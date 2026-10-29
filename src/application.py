
from answer.answer_retrieval import create_rag_chain
from store.vector_store import create_vector_store

if __name__ == "__main__":
    # Create the retriever and RAG chain
    retriever = create_vector_store()
    rag_chain = create_rag_chain(retriever)

    print("Employee Policy Chatbot")
    print("Type 'exit' to quit.")

    while True:
        query = input("\nEnter your question: ").strip()

        if query.lower() == "exit":
            print("Goodbye!")
            break

        if not query:
            print("Please enter a question.")
            continue

        # Retrieve relevant chunks for debugging
        retrieved_chunks = retriever.invoke(query)

        print("\nRetrieved chunks:")
        for i, chunk in enumerate(retrieved_chunks, start=1):
            print(f"\n--- Chunk {i} ---")
            print("Text:", chunk.page_content)
            print("Source:", chunk.metadata.get("source"))
            print("Page:", chunk.metadata.get("page_number"))

        # Generate the answer
        answer = rag_chain.invoke(query)

        print("\nAnswer:")
        print(answer)
