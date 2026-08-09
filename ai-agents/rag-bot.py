import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pypdf import PdfReader

load_dotenv()

PDF_PATH = Path("sample.pdf")
CHROMA_DIR = Path(".chroma_db")
EMBED_MODEL = "nomic-embed-text"
K = 4
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

DEEPSEEK_BASE_URL = "https://opencode.ai/zen/go/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-v4-flash"

embeddings = OllamaEmbeddings(model=EMBED_MODEL)

vectorstore = Chroma(
    collection_name="rag_bot",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR),
)


@dataclass
class RAGState:
    question: str
    documents: list[Document] = field(default_factory=list)
    answer: str = ""


def ask_deepseek(context: str, question: str) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY not set. Run: export DEEPSEEK_API_KEY=sk-..."
        )
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant answering questions strictly from the "
                    "provided context. If the context does not contain the answer, say "
                    "you don't know."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        "stream": False,
    }
    url = (
        DEEPSEEK_BASE_URL
        if DEEPSEEK_BASE_URL.endswith("/chat/completions")
        else f"{DEEPSEEK_BASE_URL}/chat/completions"
    )
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(str(path))
    return [
        Document(page_content=page.extract_text() or "", metadata={"source": str(path), "page": i + 1})
        for i, page in enumerate(reader.pages)
    ]


def ingest_pdf(path: Path) -> None:
    raw = load_pdf(path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(raw)
    vectorstore.add_documents(chunks)
    print(f"Ingested {path.name}: {len(raw)} pages -> {len(chunks)} chunks")


def retrieve(state: RAGState) -> dict:
    docs = vectorstore.similarity_search(state.question, k=K)
    return {"documents": docs}


def generate(state: RAGState) -> dict:
    context = "\n\n".join(f"[p{d.metadata['page']}] {d.page_content}" for d in state.documents)
    answer = ask_deepseek(context, state.question)
    return {"answer": answer}


graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
app = graph.compile()


def main() -> None:
    pdf = PDF_PATH if PDF_PATH.exists() else next(Path(".").glob("*.pdf"), None) or next(
        Path(".").glob("**/*.pdf"), None
    )
    if pdf is None:
        print(f"No PDF found. Put one at {PDF_PATH} (or anywhere in the project dir) and rerun.")
        sys.exit(1)

    existing = vectorstore.get(limit=1)["ids"]
    if not existing:
        ingest_pdf(pdf)
    else:
        print(f"Using existing index in {CHROMA_DIR}")

    print("RAG ready. Ask a question (type 'exit' to quit).")
    while (question := input("> ").strip()) != "exit":
        if not question:
            continue
        result = app.invoke({"question": question})
        print(f"\n{result['answer']}\n")


if __name__ == "__main__":
    main()
