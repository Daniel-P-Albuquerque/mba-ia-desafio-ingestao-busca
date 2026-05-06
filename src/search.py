import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_postgres import PGVector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "pdf_documents")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def _build_embeddings():
    if LLM_PROVIDER == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
        return GoogleGenerativeAIEmbeddings(model=model)

    from langchain_openai import OpenAIEmbeddings
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model)


def _build_llm():
    if LLM_PROVIDER == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite")
        return ChatGoogleGenerativeAI(model=model, temperature=0)

    from langchain_openai import ChatOpenAI
    model = os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano")
    kwargs = {"model": model}
    if not model.startswith("gpt-5"):
        kwargs["temperature"] = 0
    return ChatOpenAI(**kwargs)


def search_prompt(question=None):
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL não definido no .env")

    store = PGVector(
        embeddings=_build_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = _build_llm()
    inner = prompt | llm | StrOutputParser()

    def answer(q: str) -> str:
        pairs = store.similarity_search_with_score(q, k=10)
        contexto = "\n\n".join(doc.page_content for doc, _score in pairs)
        return inner.invoke({"contexto": contexto, "pergunta": q}).strip()

    if question is not None:
        return answer(question)
    return answer
