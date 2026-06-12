from flask import Flask, render_template, request, redirect
from pathlib import Path
from pypdf import PdfReader
from ollama import chat

app = Flask(__name__)

UPLOAD_FOLDER = Path("docs")
UPLOAD_FOLDER.mkdir(exist_ok=True)

documents = {}
chat_history = []


def load_documents():

    docs = {}

    for file in UPLOAD_FOLDER.iterdir():

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

        docs[file.name] = content

    return docs


documents = load_documents()


SYSTEM_PROMPT = """
You are SynAccel Assistant.

You are a professional but friendly AI assistant.

Your role is to help employees with:
- Company policies
- Procedures
- Documentation
- Security questions
- General questions

If company documentation is provided, use it.

If company documentation is not relevant, answer normally.

Be conversational and approachable.

Do not constantly mention documents unless they are actually relevant.
"""


@app.route("/", methods=["GET", "POST"])
def home():

    global documents
    global chat_history

    answer = None
    source = None
    question = None

    if request.method == "POST":

        question = request.form.get("question")

        if question:

            best_doc = None
            best_score = 0

            question_words = question.lower().split()

            for filename, content in documents.items():

                score = 0

                for word in question_words:

                    if len(word) > 3 and word in content.lower():
                        score += 1

                if score > best_score:
                    best_score = score
                    best_doc = filename

            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                }
            ]

            messages.extend(chat_history[-4:])

            if best_doc and best_score >= 3:

                messages.append(
                    {
                        "role": "user",
                        "content": f"""
Use this document if it is helpful.

DOCUMENT NAME:
{best_doc}

DOCUMENT:
{documents[best_doc][:2000]}

QUESTION:
{question}
"""
                    }
                )

                source = best_doc

            else:

                messages.append(
                    {
                        "role": "user",
                        "content": question
                    }
                )

                source = "General Assistant"

            response = chat(
                model="qwen3",
                messages=messages
            )

            answer = response.message.content

            chat_history.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            chat_history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

    return render_template(
        "index.html",
        answer=answer,
        source=source,
        question=question,
        document_count=len(documents),
        document_names=documents.keys(),
        chat_history=chat_history
    )


@app.route("/upload", methods=["POST"])
def upload():

    global documents

    uploaded_file = request.files.get("file")

    if uploaded_file and uploaded_file.filename:

        save_path = UPLOAD_FOLDER / uploaded_file.filename

        uploaded_file.save(save_path)

        documents = load_documents()

    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)
