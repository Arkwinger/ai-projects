# Private Company AI

Private Company AI is a local AI knowledge platform designed for organizations that want to interact with internal documents using a self-hosted large language model (LLM).

The project currently uses Ollama and Qwen to answer questions about uploaded documents while keeping all processing local to the user's environment.

## Current Features

Current Features:
- Local LLM via Ollama
- PDF and text document ingestion
- Multi-document support
- Basic document retrieval
- Source citations
- Question answering

## Current Workflow

```text
Documents
    ↓
Document Retrieval
    ↓
Qwen (Local LLM)
    ↓
Answer
```

## Current Status

This project is in the early prototype stage.

The current implementation can:

- Load PDF and text documents
- Extract document contents
- Retrieve relevant documents based on user questions
- Generate answers using a locally hosted LLM

## Roadmap

### Next Milestones

- Source citations
- Improved document retrieval
- Support for larger document collections
- Web-based interface

### Security Features

- Role-based access controls
- Prompt injection testing
- AI security assessment engine
- Audit logging

## Long-Term Goal

Build a secure enterprise knowledge platform that allows organizations to query internal documentation using local AI models while maintaining privacy, security, and control over company data.

## Technology Stack

- Python
- Ollama
- Qwen
- PyPDF

## Getting Started

### Requirements

- Python 3.14+
- Ollama
- Qwen model installed locally

### Installation

```bash
pip install -r requirements.txt
```

### Run

```bash
python chat.py
```

### Example Questions

```text
How many vacation days do employees receive?

Who handles password resets?

What are the six core functions of the NIST Cybersecurity Framework?
```
