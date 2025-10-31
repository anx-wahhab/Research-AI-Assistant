from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://openrouter.ai/api/v1"


def setup_llm(model_name: str) -> ChatOpenAI:
    """Sets up the LLM using OpenRouter API (compatible with LangChain v1.0.3)."""
    return ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=BASE_URL,
        model=model_name,  # e.g., "mistralai/mistral-7b-instruct"
        temperature=0.7,
        max_tokens=512,
    )


def generate_summary_sections(llm: Any, full_text: str):
    """
    Generates summary sections using the LLM.

    :param llm: The LLM instance.
    :param full_text: The full text of the paper.
    :return: A dictionary with summary sections.
    """
    prompt_template = """
    Analyze the following research paper content and extract the key sections.
    Provide a concise summary for each:

    - Abstract: Summarize the abstract.
    - Methodology: Summarize the methods used.
    - Results: Summarize the key findings.
    - Conclusions: Summarize the conclusions and implications.

    If a section is not present, state "Section not found."

    Paper content:
    {text}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"text": full_text})

    return response
    # print(response)
    # sections = {'abstract': '', 'methodology': '', 'results': '', 'conclusions': ''}
    # current_section = None
    #
    # for line in response.splitlines():
    #     line = line.strip()
    #     if line.startswith('- Abstract:'):
    #         current_section = 'abstract'
    #     elif line.startswith('- Methodology:'):
    #         current_section = 'methodology'
    #     elif line.startswith('- Results:'):
    #         current_section = 'results'
    #     elif line.startswith('- Conclusions:'):
    #         current_section = 'conclusions'
    #     elif current_section:
    #         sections[current_section] += line + " "
    #
    # for key in sections:
    #     sections[key] = sections[key].strip()

    # return sections
