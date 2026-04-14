from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small",
    encode_kwargs={"normalize_embeddings": True},
)


def _carregar_arquivo(path: Path) -> list:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loaded = PyPDFLoader(str(path)).load()
    elif suffix == ".md":
        loaded = TextLoader(str(path), encoding="utf-8").load()
    else:
        return []
    for doc in loaded:
        doc.metadata["filename"] = path.name
        doc.metadata["source"] = str(path)
    return loaded


def indexar() -> None:
    docs: list = []
    seen: set[Path] = set()

    for pattern in ("*.pdf", "*.md"):
        for path in DOCS_DIR.rglob(pattern):
            path = path.resolve()
            if path in seen:
                continue
            seen.add(path)
            docs.extend(_carregar_arquivo(path))

    if not docs:
        raise FileNotFoundError(
            f"Nenhum PDF ou Markdown (*.md) encontrado em '{DOCS_DIR}'. "
            "Coloque sua base de conhecimento nessa pasta."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )
    chunks = splitter.split_documents(docs)

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(VECTORSTORE_DIR))
    print(f"Indexados {len(chunks)} chunks em '{VECTORSTORE_DIR}'.")


def carregar_vectorstore() -> FAISS:
    if not VECTORSTORE_DIR.exists():
        indexar()
    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


if __name__ == "__main__":
    indexar()