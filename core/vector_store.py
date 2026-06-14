from typing import IO

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_unstructured import UnstructuredLoader

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True},
)

vector_store = Chroma(
    collection_name="init_collection",
    embedding_function=embeddings,
    persist_directory="data/chroma_langchain_db",  # Where to save data locally, remove if not necessary
)


def load_pdf(file: IO[bytes], file_name: str):
    """adds the file bytes to chromadb"""

    # TODO refine these
    loader = UnstructuredLoader(
        file=file,
        metadata_filename=file_name,
        chunking_strategy="by_title",
        max_characters=1500,
        new_after_n_chars=1000,
        overlap=100,
    )

    docs = loader.load()
    vector_store.add_documents(documents=docs)


def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
