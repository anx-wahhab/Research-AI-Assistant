from typing import Dict
from utils.helpers import load_pdf
from utils.llm_setup import setup_llm, generate_summary_sections
from utils.exporters import export_to_word

class ResearchAssistant:
    """
    A LangChain-based research assistant that analyzes uploaded research papers
    using DeepSeek V3 Base via OpenRouter API. It generates structured Word summaries
    (abstract, methodology, results, conclusions) and allows querying for specific details.
    """

    def __init__(self, pdf_path: str, model_name: str = "deepseek/deepseek-chat"):
        """
        Initializes the ResearchAssistant.

        :param pdf_path: Path to the uploaded PDF research paper.
        :param model_name: The model to use (default: deepseek/deepseek-chat).
        """
        self.llm = setup_llm(model_name)
        self.documents = load_pdf(pdf_path)

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