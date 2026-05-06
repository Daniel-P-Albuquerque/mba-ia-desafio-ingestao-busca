# Desafio MBA Engenharia de Software com IA — Full Cycle

Ingestão e busca semântica em PDF usando **LangChain**, **PostgreSQL + pgVector** e CLI em Python. Suporta tanto **OpenAI** quanto **Google Gemini**, escolhidos via variável de ambiente.

---

## Pré-requisitos

- Python 3.11+
- Docker e Docker Compose
- Uma chave de API de **um** dos provedores:
  - OpenAI (`OPENAI_API_KEY`), ou
  - Google AI Studio (`GOOGLE_API_KEY`)

---

## Setup

### 1. Clonar o repositório

```bash
git clone https://github.com/Daniel-P-Albuquerque/mba-ia-desafio-ingestao-busca.git
cd mba-ia-desafio-ingestao-busca
```

### 2. Criar e ativar o ambiente virtual

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar variáveis de ambiente

Linux/macOS:

```bash
cp .env.example .env
```

Windows:

```powershell
copy .env.example .env
```

Edite o `.env`: defina `LLM_PROVIDER` (`openai` ou `google`) e preencha a chave correspondente.

---

## Ordem de execução

### 1. Subir o banco de dados (Postgres + pgVector)

```bash
docker compose up -d
```

O `docker-compose.yml` sobe um Postgres 17 com a extensão `vector` já criada (via service de bootstrap).

### 2. Ingerir o PDF

A partir da **raiz do repositório**:

```bash
python src/ingest.py
```

O script:

- Carrega `document.pdf`
- Particiona em chunks de **1000 caracteres com overlap de 150**
- Gera embeddings via o provedor configurado em `LLM_PROVIDER`
- Persiste tudo na coleção `PG_VECTOR_COLLECTION_NAME` no Postgres
- **Recria a coleção do zero a cada execução** (idempotente)

Saída esperada:

```text
Ingested N chunks into 'pdf_documents' (provider=openai).
```

### 3. Conversar com o PDF (CLI)

A partir da **raiz do repositório**:

```bash
python src/chat.py
```

Exemplo de uso:

```text
Faça sua pergunta. (Pressione Enter vazio ou digite 'sair' para encerrar.)

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

PERGUNTA: Qual a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

PERGUNTA: sair
```

Cada pergunta é vetorizada, busca os **10 chunks mais relevantes** (`similarity_search_with_score(k=10)`), monta o contexto e chama a LLM com o prompt do desafio. A LLM só responde com base no contexto encontrado; senão devolve a frase padrão.

---

## Variáveis de ambiente

| Variável | Default | Descrição |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` | `openai` ou `google` |
| `OPENAI_API_KEY` | — | Chave OpenAI (se `LLM_PROVIDER=openai`) |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Modelo de embeddings da OpenAI |
| `OPENAI_LLM_MODEL` | `gpt-5-nano` | Modelo de chat da OpenAI |
| `GOOGLE_API_KEY` | — | Chave Google AI Studio (se `LLM_PROVIDER=google`) |
| `GOOGLE_EMBEDDING_MODEL` | `models/embedding-001` | Modelo de embeddings do Gemini |
| `GOOGLE_LLM_MODEL` | `gemini-2.5-flash-lite` | Modelo de chat do Gemini |
| `DATABASE_URL` | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` | URL do Postgres (driver psycopg3) |
| `PG_VECTOR_COLLECTION_NAME` | `pdf_documents` | Nome da coleção pgVector |
| `PDF_PATH` | `document.pdf` | Caminho do PDF (relativo à raiz do repo ou absoluto) |

---

## Trocando de provedor

Para alternar entre OpenAI e Gemini:

1. Edite `.env`: troque `LLM_PROVIDER` e preencha a chave correspondente.
2. Re-execute a ingestão (`python src/ingest.py`).

Embeddings de provedores diferentes são incompatíveis (dimensões e espaços vetoriais distintos), então a coleção precisa ser regerada ao trocar.

---

## Notas

- **`gpt-5-nano`** rejeita `temperature` diferente de `1` na API atual da OpenAI. O código omite esse parâmetro automaticamente para qualquer modelo cujo nome comece com `gpt-5`.
- A `DATABASE_URL` usa o driver **psycopg3** (`postgresql+psycopg://...`). Não troque para `postgresql://...` puro — o `langchain-postgres==0.0.15` falha com psycopg2.
- Se `gemini-2.5-flash-lite` der erro de modelo inexistente, troque `GOOGLE_LLM_MODEL` para `gemini-2.0-flash` (ou outro disponível na sua região).
- Para encerrar o chat: digite `sair`, `quit`, `exit`, ou pressione Enter sem texto. `Ctrl+C` também funciona.
