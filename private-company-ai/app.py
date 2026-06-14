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

# Toggle this to False for normal users
ADMIN_MODE = True

load_documents()

SYSTEM_PROMPT = """
You are SynAccel Assistant.

Provide professional, business-friendly answers.

Rules:
- Use company documentation whenever possible.
- Be clear and direct.
- Match the length of the answer to the question.
- Short questions should receive concise answers.
- Complex questions can receive detailed answers.
- Prefer bullet points when listing information.
- Avoid unnecessary disclaimers.
- Avoid filler phrases.
- Do not use emojis.

If relevant company documentation is provided:
- Use it as your primary source.
- Base your answer on the documentation.

If no documentation is provided:
- Answer naturally using your own knowledge.
- Maintain a professional tone.
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

            # No useful documentation found
            if not context.strip():

                response = chat(
                    model="qwen3",
                    messages=[
                        {
                            "role": "system",
                            "content": """
You are SynAccel Assistant.

You are a professional AI assistant.

You can:
- Answer questions using company documentation.
- Answer general cybersecurity questions.
- Hold natural conversations.
- Help users with technology questions.

If documentation is unavailable,
answer the user's question normally.
"""
                        },
                        {
                            "role": "user",
                            "content": question
                        }
                    ]
                )

                answer = response.message.content
                source = None

            else:

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
                    source = None

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

            # Keep only recent messages
            chat_history = chat_history[-20:]

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
        chat_history=chat_history,
        admin_mode=ADMIN_MODE
    )


@app.route("/upload", methods=["POST"])
def upload():

    uploaded_file = request.files.get("file")

    if uploaded_file and uploaded_file.filename:

        save_path = UPLOAD_FOLDER / uploaded_file.filename

        uploaded_file.save(save_path)

        load_documents()

    return redirect("/")


@app.route("/delete/<filename>", methods=["POST"])
def delete_document(filename):

    file_path = UPLOAD_FOLDER / filename

    if file_path.exists():

        file_path.unlink()

        load_documents()

    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)
