from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

PDF_PATH = Path("src/data/fictional_employee_policies.pdf")


def load_pdf():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found at: {PDF_PATH.resolve()}")

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    if not documents:
        raise ValueError("No pages were extracted from the PDF.")

    # Preserve the source and add human-readable page numbers
    for document in documents:
        document.metadata["page_number"] = document.metadata["page"] + 1

    print(f"Total pages loaded: {len(documents)}")

    for document in documents:
        print("\nMetadata:", document.metadata)
        print("Text:", document.page_content)

    return documents
