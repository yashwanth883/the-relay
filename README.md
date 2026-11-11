# The Relay: Research Assistant Pipeline & Benchmark Harness

A production-grade, rate-limit resilient question-answering assistant and load-testing harness designed for the Vaani ML Engineering Intern assignment.

The system queries a 3-paper library on arXiv:
1. **"Attention Is All You Need"** (`arXiv:1706.03762v7`)
2. **"Moshi: a speech-text foundation model for real-time dialogue"** (`arXiv:2410.00037v1`)
3. **"Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"** (`arXiv:2507.07955v2`)

---

## Repository Structure

```text
.
├── download_papers.py      # Downloads the exact arXiv PDF versions and extracts text by page
├── writeup.md              # The full 2-page engineering write-up with all required sections and tables
├── writeup.html            # Clean HTML format of the write-up
├── conversation_history.md # Complete transcript of prompts and interactions with the assistant
├── results.csv             # Raw per-request test log (158 requests across all load & retrieval runs)
├── README.md               # Overview, run instructions, and reproducibility guide
├── data/
│   ├── papers/             # Downloaded official arXiv PDFs
│   └── extracted/          # Clean text extractions demarcated by page number
└── src/
    ├── __init__.py
    ├── indexer.py          # BM25 + phrase & section reranker supporting settings A, B, and C
    ├── standin_service.py  # High-fidelity Groq free-tier simulator (RPM, TPM, RPD, degradation, streaming)
    ├── client.py           # Unified model client with retry policy, backoff jitter, and nanosecond timers
    ├── pipeline.py         # 5-step agentic pipeline with tool calls, branch (refusal), loop, and verification
    ├── eval_questions.py   # 18 curated domain questions (factual, cross-paper, and unanswerable)
    ├── load_tester.py      # Automated benchmark harness for retrieval settings & 5 load shapes
    └── compute_tables.py   # Strictly recomputes Tables 1, 2a, 2b, and 3 directly from results.csv
```

---

## Quickstart & Reproducibility

### 1. Requirements
Ensure Python 3.10+ is installed. Dependencies include:
```bash
pip install pypdf numpy pandas httpx
```

### 2. Download and Extract the Papers
```bash
python download_papers.py
```

### 3. Re-run the Full Benchmark Suite
To re-run all 5 load scenarios and the retrieval settings comparison to generate a fresh `results.csv`:
```bash
python -m src.load_tester
```
*(Optional: Set `export GROQ_API_KEY="your-key"` to run steady load against real hosted Groq models)*.

### 4. Recompute the Tables
Reviewers can verify and recompute every figure in Tables 2a, 2b, and 3 directly from `results.csv`:
```bash
python -m src.compute_tables
```

---

## Submission Checklist
- [x] **writeup.md**: 2 pages, all required headings in exact order, fully filled tables, reflection, and Vaani feedback.
- [x] **results.csv**: Exact column format, 1 row per request, all table figures strictly recomputable.
- [x] **Source Code**: Modular Python implementation of pipeline, stand-in service, indexer, and test harness.
- [x] **conversation_history.md**: Complete prompt and conversation transcript with the AI assistant.

