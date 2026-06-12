from pypdf import PdfReader
from ollama import chat

reader = PdfReader("NIST.CSWP.29.pdf")

pdf_text = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        pdf_text += text + "\n"

question = input("Ask a question about the PDF: ")

prompt = f"""
Use the following document to answer the question.

DOCUMENT:
{pdf_text[:50000]}

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
