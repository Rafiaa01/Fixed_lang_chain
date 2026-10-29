from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = Path("src/data/fictional_employee_policies.pdf")


def load_pdf():
    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    for document in documents:
        document.metadata["page_number"] = document.metadata["page"] + 1

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    return chunks


def create_chunks():
    documents = load_pdf()
    chunks = split_documents(documents)

    print("\n===== ALL CREATED CHUNKS =====")
    print("Total chunks:", len(chunks))

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {i} ---")
        print("Text:")
        print(chunk.page_content)

        print("Metadata:")
        for key, value in chunk.metadata.items():
            print(f"  {key}: {value}")

        print("-" * 60)

    return chunks


