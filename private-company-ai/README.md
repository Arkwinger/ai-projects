# SynAccel Assistant

SynAccel Assistant is a local Retrieval-Augmented Generation (RAG) platform designed for internal company environments.

The platform combines a self-hosted Large Language Model (LLM) with semantic document retrieval to provide employees with a conversational assistant capable of answering both general questions and organization-specific questions.

All processing occurs locally through Ollama, allowing organizations to explore AI-assisted knowledge management without sending sensitive data to external providers.

---

## Current Features

- Local LLM powered by Ollama
- Qwen model integration
- Retrieval-Augmented Generation (RAG)
- ChromaDB vector database
- Sentence Transformers semantic search
- PDF document ingestion
- TXT document ingestion
- Dynamic document uploads
- Document deletion and re-indexing
- Conversational memory
- Source attribution
- Multi-document knowledge base
- Semantic similarity search
- SynAccel-branded web interface
- Local-first architecture
- No external API dependencies

---

## Example Use Cases

- Employee onboarding assistance
- Internal policy lookup
- Security documentation support
- Procedure and workflow guidance
- Knowledge management
- Internal research assistant
- Cybersecurity reference assistant
- Technical documentation search
- Internal compliance reference
- AI-powered knowledge retrieval

---

## Current Workflow

```text
User Question
      ↓
SynAccel Assistant
      ↓
Conversation Context
      ↓
Semantic Retrieval
      ↓
ChromaDB Vector Search
      ↓
Relevant Documentation
      ↓
Qwen via Ollama
      ↓
Response + Source Attribution
```

---

## Project Structure

```text
AIProject/
│
├── app.py
├── retrieval.py
├── chat.py
├── requirements.txt
│
├── docs/
│   ├── handbook.txt
│   ├── policy.pdf
│   └── company_documents.pdf
│
├── chroma_db/
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── logo.png
│
└── screenshots/
```

---

## Technology Stack

- Python
- Flask
- Ollama
- Qwen
- ChromaDB
- Sentence Transformers
- PyPDF
- HTML
- CSS

---

## Retrieval Architecture

SynAccel Assistant uses semantic retrieval rather than traditional keyword search.

### Process

1. Documents are uploaded to the knowledge base
2. Documents are split into chunks
3. Sentence Transformers generate vector embeddings
4. Embeddings are stored in ChromaDB
5. User questions are converted into embeddings
6. Similar document chunks are retrieved
7. Retrieved context is sent to Qwen
8. Responses are generated using retrieved knowledge

This approach allows the assistant to locate relevant information even when the user's wording differs from the original document wording.

---

## Why Local AI?

Many organizations are interested in AI-assisted knowledge management but cannot upload sensitive internal documents to third-party cloud services.

This project explores a local-first approach where:

- Internal documents remain on company systems
- AI inference occurs locally through Ollama
- Knowledge retrieval is performed without external APIs
- Organizations maintain control of their data
- Sensitive information never leaves the environment

---

## Current Status

### Functional RAG Knowledge Assistant

Current capabilities include:

- Semantic document retrieval
- Conversational interactions
- Context retention
- Local inference
- Source attribution
- Multi-document knowledge search
- Dynamic document management
- Internal knowledge retrieval

---

## Future Development

Potential future enhancements include:

- Streaming responses
- Authentication and user accounts
- Role-based document access
- Audit logging
- Prompt injection testing
- AI security assessments
- Enterprise deployment options
- API integrations
- Multi-user support
- Advanced analytics
- Document versioning

---

## Security Considerations

SynAccel Assistant is designed around a local-first model.

Benefits include:

- No external AI API dependency
- Internal documents remain local
- Reduced data exposure risk
- Greater control over AI infrastructure
- Suitable for research and enterprise environments

---

## License

This project is intended for research, educational, and internal enterprise AI experimentation.
