import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "pdf_documents")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()


def _resolve_pdf_path(pdf_path: str) -> Path:
    p = Path(pdf_path)
    if not p.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent
        p = repo_root / p
    return p


def _build_embeddings():
    if LLM_PROVIDER == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
        return GoogleGenerativeAIEmbeddings(model=model)

    from langchain_openai import OpenAIEmbeddings
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model)


def ingest_pdf():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL não definido no .env")

    pdf_path = _resolve_pdf_path(PDF_PATH)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF não encontrado em {pdf_path}")

    docs = PyPDFLoader(str(pdf_path)).load()

    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=False,
    ).split_documents(docs)

    if not splits:
        raise RuntimeError("O PDF não gerou nenhum chunk após o split.")

    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)},
        )
        for d in splits
    ]
    ids = [f"doc-{i}" for i in range(len(enriched))]

    store = PGVector(
        embeddings=_build_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
        pre_delete_collection=True,
    )
    store.add_documents(documents=enriched, ids=ids)

    print(f"Ingested {len(enriched)} chunks into '{COLLECTION_NAME}' (provider={LLM_PROVIDER}).")


if __name__ == "__main__":
    ingest_pdf()
