from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from typing import Dict, Any
import os

BASE_URL = "https://openrouter.ai/api/v1"

def setup_llm(openrouter_api_key: str, model_name: str) -> ChatOpenAI:
    """Sets up the LLM using OpenRouter API."""
    return \
        ChatOpenAI(
    api_key=os.getenv('OPENROUTER_API_KEY'),
    base_url=BASE_URL,
    model="deepseek/deepseek-chat",  # or any Mistral model
    temperature=0.7,
    max_tokens=512,

    )


def setup_qa_chain(llm: Any, vectorstore: FAISS) -> RetrievalQA:
    """Sets up the RetrievalQA chain for querying."""
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
    )


def generate_summary_sections(llm: Any, full_text: str) -> Dict[str, str]:
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
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"text": full_text})

    sections = {'abstract': '', 'methodology': '', 'results': '', 'conclusions': ''}
    current_section = None
    for line in response.splitlines():
        line = line.strip()
        if line.startswith('- Abstract:'):
            current_section = 'abstract'
        elif line.startswith('- Methodology:'):
            current_section = 'methodology'
        elif line.startswith('- Results:'):
            current_section = 'results'
        elif line.startswith('- Conclusions:'):
            current_section = 'conclusions'
        elif current_section:
            sections[current_section] += line + " "

    for key in sections:
        sections[key] = sections[key].strip() or "Section not found."

    return sections