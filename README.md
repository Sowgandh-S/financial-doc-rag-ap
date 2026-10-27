# Financial Document Q&A API

A learning project that searches permitted financial PDFs and answers questions using retrieved passages. It uses local Ollama models for embeddings and generation, and persistent ChromaDB storage.

> **Educational use only. This project is not financial, investment, tax, or legal advice.**

## Features

- Upload text-based PDF documents.
- Extract text and preserve filename and page number.
- Split text into overlapping chunks.
- Generate embeddings locally using Ollama.
- Store vectors and passages persistently in ChromaDB.
- Retrieve relevant passages for a question.
- Generate answers from the retrieved excerpts.
- Return source filenames, page numbers, and passages.
- Avoid answering when no relevant passages are found.

## Requirements

- Python 3.10 or newer.
- [Ollama](https://ollama.com/) installed and running.
- Sufficient memory and disk space for local models.
- Documents you are permitted to use.

## Install local models

Start Ollama, then run:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Install the API

```bash
git clone https://github.com/YOUR-USERNAME/financial-doc-rag-api.git
cd financial-doc-rag-api
python -m venv .venv
```

Activate the environment.

Linux or macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Make sure Ollama is running, then start the API:

```bash
uvicorn app.main:app --reload
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

## Use

1. Call `POST /documents` and upload a text-based PDF.
2. Call `POST /questions` with a JSON body such as:

```json
{
  "question": "What risks does the report discuss?"
}
```

3. Review the answer and source excerpts. Verify every answer against the cited document.

## Persistent storage

ChromaDB stores its local index in `chroma_db/`. The directory is created automatically and should persist across server restarts. It is local application data and should not be committed to GitHub.

## Limitations

- The project is an educational prototype.
- Scanned PDFs need OCR, which is not included.
- The vector index is local and has no multi-user access control.
- Retrieval settings have not been benchmarked.
- Generated answers may be incorrect or incomplete.
- The app does not verify accounting data or investment claims.
- It does not provide financial advice or recommendations.

## Privacy and document use

Use only documents you have permission to process. Avoid committing reports or other documents to a public repository unless their terms allow redistribution. The sample app processes documents locally through Ollama, but review your machine and network configuration before using sensitive files.

## Planned improvements

- Add automated tests for extraction, indexing, retrieval, and citations.
- Add document deletion and index management.
- Add OCR for scanned reports.
- Build a small evaluation set and measure retrieval quality.
- Add upload limits, authentication, and access control.
- Add Docker after local operation is verified.

## Attribution

This is an independent learning project. Credit and comply with licenses for any third-party code, data, or assets used.

## Resume description

Only after implementation and testing, you may describe the work accurately, for example:

> Built a local financial-document Q&A prototype with PDF text extraction, persistent ChromaDB retrieval, and source-referenced answers.

Mention only technologies and results you personally implemented and verified.
