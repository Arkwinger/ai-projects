from pathlib import Path
from pypdf import PdfReader
from ollama import chat

documents = {}

docs_folder = Path("docs")

# Load all documents
for file in docs_folder.iterdir():

    content = ""

    if file.suffix.lower() == ".pdf":

        reader = PdfReader(file)

        for page in reader.pages:
            text = page.extract_text()

            if text:
                content += text + "\n"

    elif file.suffix.lower() == ".txt":

        with open(file, "r", encoding="utf-8") as f:
            content = f.read()

    documents[file.name] = content


question = input("Ask a question: ")

# Basic retrieval
best_doc = None
best_score = 0

question_words = question.lower().split()

for filename, content in documents.items():

    score = 0
    content_lower = content.lower()

    for word in question_words:
        if word in content_lower:
            score += 1

    if score > best_score:
        best_score = score
        best_doc = filename


if best_doc:

    print(f"\nUsing document: {best_doc}")

    prompt = f"""
Use the following document to answer the question.

DOCUMENT NAME:
{best_doc}

DOCUMENT:
{documents[best_doc][:50000]}

QUESTION:
{question}
"""

else:

    prompt = f"""
Answer the question as best you can.

QUESTION:
{question}
"""


response = chat(
    model="qwen3",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nAnswer:")
print(response.message.content)

if best_doc:
    print(f"\nSource: {best_doc}")
