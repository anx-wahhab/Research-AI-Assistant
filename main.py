import os
from core.assisstant import ResearchAssistant

# Example usage
if __name__ == "__main__":
    # Replace with your actual API key and PDF path
    api_key = os.getenv("OPENROUTER_API_KEY")  # Or hardcode for testing
    pdf_path = "research_paper.pdf"
    output_word_path = "summary.docx"
    model_name = "deepseek/deepseek-chat"  # Assuming V3 base availability

    assistant = ResearchAssistant(pdf_path=pdf_path, model_name=model_name)

    # Generate and export summary
    summary = assistant.generate_summary()
    assistant.export_summary_to_word(summary, output_word_path)