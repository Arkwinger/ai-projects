from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb


model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="documents"
)


def load_documents():

    docs_folder = Path("docs")

    collection.delete(
        where={}
    )

    doc_id = 0

    for file in docs_folder.iterdir():

        content = ""

        if file.suffix.lower() == ".pdf":

            try:

                reader = PdfReader(file)

                for page in reader.pages:

                    text = page.extract_text()

                    if text:
                        content += text + "\n"

            except Exception:
                continue

        elif file.suffix.lower() == ".txt":

            try:

                with open(file, "r", encoding="utf-8") as f:
                    content = f.read()

            except Exception:
                continue

        chunks = [
            content[i:i + 1000]
            for i in range(0, len(content), 1000)
        ]

        for chunk in chunks:

            embedding = model.encode(chunk).tolist()

            collection.add(
                ids=[str(doc_id)],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[
                    {
                        "source": file.name
                    }
                ]
            )

            doc_id += 1


def search_documents(query):

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    return results
