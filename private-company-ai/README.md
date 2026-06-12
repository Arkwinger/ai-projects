# SynAccel Assistant

SynAccel Assistant is a local AI knowledge assistant designed for internal company environments.

The platform combines a self-hosted large language model (LLM) with internal documentation to provide employees with a conversational assistant capable of answering both general questions and organization-specific questions.

All processing occurs locally through Ollama, allowing organizations to explore AI-assisted knowledge management without sending sensitive data to external providers.

---

## Current Features

* Local LLM powered by Ollama
* Qwen integration
* PDF document ingestion
* Text document ingestion
* Conversational assistant interface
* Conversation memory
* Multi-document knowledge base
* Basic document retrieval
* Source attribution
* SynAccel-branded web interface
* Local-first architecture

---

## Example Use Cases

* Employee onboarding assistance
* Internal policy lookup
* Security documentation support
* Procedure and workflow guidance
* Knowledge management
* Internal research assistant
* Cybersecurity reference assistant

---

## Current Workflow

```text
User Question
      ↓
SynAccel Assistant
      ↓
Conversation Context
      ↓
Document Retrieval (if relevant)
      ↓
Qwen via Ollama
      ↓
Response
```

---

## Project Structure

```text
private-company-ai/
│
├── app.py
├── chat.py
├── requirements.txt
│
├── docs/
│   ├── handbook.txt
│   ├── it_policy.txt
│   └── company_documents.pdf
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

* Python
* Flask
* Ollama
* Qwen
* PyPDF
* HTML
* CSS

---

## Why Local AI?

Many organizations are interested in AI-assisted knowledge management but cannot upload sensitive internal documents to third-party cloud services.

This project explores a local-first approach where:

* Internal documents remain on company systems
* AI inference occurs locally through Ollama
* Knowledge retrieval is performed without external APIs
* Organizations maintain control of their data

---

## Current Status

Research Prototype

The current version supports:

* Conversational interactions
* Context retention
* Local inference
* Document-aware responses
* Internal knowledge retrieval

Future development will focus on:

* Semantic retrieval
* Streaming responses
* Role-based access controls
* Prompt injection testing
* AI security assessments
* Audit logging
* Enterprise deployment options

---


