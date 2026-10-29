from chunking.recursive_chunking import create_chunks
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def create_vector_store():
    chunks = create_chunks()

    if not chunks:
        raise ValueError("No chunks found. Check your PDF and chunking step.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    print("FAISS index created successfully.")
    print("Total indexed chunks:", vector_store.index.ntotal)

    # Inspect the first embedding vector
    embedding_vector = vector_store.index.reconstruct(0)

    print("\nFirst embedding vector:")
    print(embedding_vector)
    print("Vector dimensions:", len(embedding_vector))

    # Save the index
    vector_store.save_local("faiss_index")

    # Create a retriever that returns the top 3 chunks
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    return retriever
