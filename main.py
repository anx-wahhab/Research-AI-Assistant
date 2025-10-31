import os
from core.assisstant import ResearchAssistant

# Example usage
if __name__ == "__main__":
    api_key = os.getenv("OPENROUTER_API_KEY")
    pdf_path = "research_paper.pdf"
    output_word_path = "summary.docx"
    model_name = "deepseek/deepseek-chat"

    assistant = ResearchAssistant(openrouter_api_key=api_key, pdf_path=pdf_path, model_name=model_name)

    # Generate
    summary = assistant.generate_summary()

    # Example query
    query = "What is the main hypothesis of the paper?"
    response = assistant.query_paper(query)
    print("Query Response:", response)