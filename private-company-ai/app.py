from flask import Flask, render_template, request, redirect
from pathlib import Path
from ollama import chat

from retrieval import (
    load_documents,
    search_documents
)

app = Flask(__name__)

UPLOAD_FOLDER = Path("docs")
UPLOAD_FOLDER.mkdir(exist_ok=True)

chat_history = []

load_documents()

SYSTEM_PROMPT = """
You are SynAccel Assistant.

You are a professional but friendly AI assistant.

Your role is to help employees with:
- Company policies
- Procedures
- Documentation
- Security questions
- General questions

Use company documentation whenever relevant.

If the documentation does not contain the answer,
say so clearly.

Be conversational and approachable.
"""


@app.route("/", methods=["GET", "POST"])
def home():

    global chat_history

    answer = None
    source = None
    question = None

    if request.method == "POST":

        question = request.form.get("question")

        if question:

            results = search_documents(question)

            context = ""

            sources = set()

            if results and results["documents"]:

                for doc_group, meta_group in zip(
                    results["documents"],
                    results["metadatas"]
                ):

                    for doc, meta in zip(doc_group, meta_group):

                        context += doc + "\n\n"

                        if meta and "source" in meta:
                            sources.add(meta["source"])

            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                }
            ]

            messages.extend(chat_history[-4:])

            messages.append(
                {
                    "role": "user",
                    "content": f"""
Use the following company documentation to answer the question.

DOCUMENTATION:
{context}

QUESTION:
{question}
"""
                }
            )

            response = chat(
                model="qwen3",
                messages=messages
            )

            answer = response.message.content

            if sources:
                source = ", ".join(sorted(sources))
            else:
                source = "No source found"

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

    document_names = []

    for file in UPLOAD_FOLDER.iterdir():
        document_names.append(file.name)

    return render_template(
        "index.html",
        answer=answer,
        source=source,
        question=question,
        document_count=len(document_names),
        document_names=document_names,
        chat_history=chat_history
    )


@app.route("/upload", methods=["POST"])
def upload():

    uploaded_file = request.files.get("file")

    if uploaded_file and uploaded_file.filename:

        save_path = UPLOAD_FOLDER / uploaded_file.filename

        uploaded_file.save(save_path)

        load_documents()

    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)
