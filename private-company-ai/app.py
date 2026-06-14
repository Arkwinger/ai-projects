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

ADMIN_MODE = True

load_documents()

SYSTEM_PROMPT = """
You are SynAccel Assistant.

Answer using the provided company documentation.

Rules:

- Use the documentation as your primary source.
- Give direct answers.
- Be concise but helpful.
- When listing items, use bullet points.
- Briefly explain the answer when useful.
- Write naturally and professionally.
- Do not use emojis.
- Do not make up information.
- Do not speculate.
- If the answer exists in the documentation, use it.
- Do not mention that you are using documentation.

If the answer cannot be found in the documentation, respond exactly with:

I could not find that information in the available documents.

Example:

Question:
What are the NIST CSF Functions?

Answer:
The NIST Cybersecurity Framework (CSF) consists of six core functions:

• Govern
• Identify
• Protect
• Detect
• Respond
• Recover

These functions help organizations manage cybersecurity risk and improve cybersecurity outcomes.
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

            if results and results.get("documents"):

                for doc_group, meta_group in zip(
                    results["documents"],
                    results["metadatas"]
                ):

                    for doc, meta in zip(doc_group, meta_group):

                        context += doc + "\n\n"

                        print("\n===== RETRIEVED DOCUMENT =====")
                        print(doc[:1000])
                        print("==============================\n")

                        if meta and "source" in meta:
                            sources.add(meta["source"])

            # ==========================
            # NORMAL CHAT MODE
            # ==========================

            if not context.strip():

                response = chat(
                    model="qwen3",
                    messages=[
                        {
                            "role": "system",
                            "content": """
You are SynAccel Assistant.

Be conversational and concise.

Rules:
- Do not use emojis.
- Do not introduce yourself unless asked.
- Keep most answers under 5 sentences.
- Respond naturally.
- Be professional but friendly.
"""
                        }
                    ]
                    + chat_history[-6:]
                    + [
                        {
                            "role": "user",
                            "content": question
                        }
                    ]
                )

                answer = response.message.content
                source = None

            # ==========================
            # DOCUMENT MODE
            # ==========================

            else:

                response = chat(
                    model="qwen3",
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": f"""
Use the following documentation to answer the question.

DOCUMENTATION:
{context}

QUESTION:
{question}
"""
                        }
                    ]
                )

                answer = response.message.content

                if sources:
                    source = ", ".join(sorted(sources))
                else:
                    source = None

            # ==========================
            # SAVE CHAT HISTORY
            # ==========================

            chat_history.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            assistant_message = answer

            if source:
                assistant_message += f"\n\nSource: {source}"

            chat_history.append(
                {
                    "role": "assistant",
                    "content": assistant_message
                }
            )

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
