from typing import Dict
from research_assistant.utils.helpers import load_pdf, create_vectorstore, split_documents
from research_assistant.llm_setup import setup_llm, setup_qa_chain, generate_summary_sections
from research_assistant.exporters import export_to_word

class ResearchAssistant:
    """
    A LangChain-based research assistant that analyzes uploaded research papers
    using DeepSeek V3 Base via OpenRouter API. It generates structured Word summaries
    (abstract, methodology, results, conclusions) and allows querying for specific details.
    """

    def __init__(self, openrouter_api_key: str, pdf_path: str, model_name: str = "deepseek/deepseek-chat"):
        """
        Initializes the ResearchAssistant.

        :param openrouter_api_key: Your OpenRouter API key.
        :param pdf_path: Path to the uploaded PDF research paper.
        :param model_name: The model to use (default: deepseek/deepseek-chat).
        """
        self.llm = setup_llm(openrouter_api_key, model_name)
        self.documents = load_pdf(pdf_path)
        self.split_docs = split_documents(self.documents)
        self.vectorstore = create_vectorstore(self.split_docs)
        self.qa_chain = setup_qa_chain(self.llm, self.vectorstore)

    def generate_summary(self) -> Dict[str, str]:
        """
        Generates a structured summary of the paper using the LLM.

        :return: A dictionary with keys: 'abstract', 'methodology', 'results', 'conclusions'.
        """
        full_text = "\n".join(doc.page_content for doc in self.documents)
        return generate_summary_sections(self.llm, full_text)

    def export_summary_to_word(self, summary: Dict[str, str], output_path: str) -> None:
        """
        Exports the structured summary to a Word document.

        :param summary: The summary dictionary.
        :param output_path: Path to save the Word file.
        """
        export_to_word(summary, output_path)

    def query_paper(self, query: str) -> str:
        """
        Queries the paper for specific details or clarifications.

        :param query: The user's query.
        :return: The response from the QA chain.
        """
        result = self.qa_chain.invoke({"query": query})
        return result['result']