# Academic Claim Verifier — One-Day MVP Implementation Plan

## 1. Project Goal

Build a small research tool that verifies whether claims in AI-generated academic writing are supported by a user-provided collection of research papers.

The core workflow is:

```text
Upload PDFs
    ↓
Extract page-level text
    ↓
Chunk text while preserving citation metadata
    ↓
Create embeddings
    ↓
Store chunks in a vector database
    ↓
Paste AI-generated academic text
    ↓
Extract atomic claims
    ↓
Retrieve evidence for each claim
    ↓
Classify each claim
    ↓
Show verdict + evidence + paper + page
```

The MVP should classify each claim as:

- `SUPPORTED`
- `PARTIALLY_SUPPORTED`
- `UNSUPPORTED`

Important semantic rule:

> `UNSUPPORTED` means "not sufficiently supported by the uploaded evidence."  
> It does **not** mean the claim is universally false.

The project should be useful to a PhD student checking AI-generated literature reviews, related-work sections, paper summaries, research notes, or draft claims against a set of source papers.

---

# 2. MVP Definition

A successful one-day MVP must allow a user to:

1. Upload multiple text-based PDF research papers.
2. Extract text while preserving:
   - filename
   - page number
   - chunk ID
3. Build a searchable semantic index over the papers.
4. Paste a paragraph of academic text.
5. Automatically decompose the paragraph into atomic claims.
6. Retrieve the most relevant paper passages for each claim.
7. Ask an LLM to evaluate whether the evidence supports the claim.
8. Display:
   - claim text
   - verdict
   - short explanation
   - supporting evidence excerpts
   - source filename
   - page number
   - retrieval similarity score if available
9. Run against a small manually labeled evaluation set.
10. Produce simple evaluation metrics.

The MVP must work locally.

---

# 3. Explicit Non-Goals

Do **not** implement the following in the first version:

- autonomous agents
- multi-agent systems
- ReAct loops
- LangChain
- LlamaIndex
- web search
- automatic paper discovery
- citation graph traversal
- OCR for scanned PDFs
- table understanding
- image understanding
- rerankers
- hybrid BM25 + vector retrieval
- knowledge graphs
- distributed databases
- production authentication
- cloud deployment
- complex async infrastructure
- fine-tuning
- model training
- full truth verification against the internet

The first version should be a deterministic pipeline.

Prefer simple, inspectable Python over framework-heavy abstractions.

---

# 4. Recommended Technology Stack

Use:

- Python 3.11+
- Streamlit for the UI
- PyMuPDF (`fitz`) for PDF extraction
- OpenAI API for:
  - embeddings
  - claim extraction
  - claim verification
- ChromaDB for local vector storage
- Pydantic for structured schemas
- pytest for testing
- python-dotenv for environment variables

Optional:

- pandas for evaluation summaries

Environment variable:

```bash
OPENAI_API_KEY=...
```

Do not hard-code secrets.

---

# 5. Repository Structure

Create the following structure:

```text
academic-claim-verifier/
│
├── app.py
├── README.md
├── plan.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── schemas.py
│   ├── pdf_parser.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── claim_extractor.py
│   ├── retriever.py
│   ├── verifier.py
│   ├── pipeline.py
│   └── evaluation.py
│
├── prompts/
│   ├── claim_extraction.txt
│   └── claim_verification.txt
│
├── data/
│   ├── papers/
│   ├── chroma/
│   └── eval/
│       └── sample_eval.json
│
└── tests/
    ├── test_pdf_parser.py
    ├── test_chunker.py
    ├── test_claim_extractor.py
    ├── test_retriever.py
    ├── test_verifier.py
    └── test_evaluation.py
```

Keep modules small and easy to inspect.

---

# 6. Core Data Models

Define all shared schemas in `src/schemas.py`.

Use Pydantic models where appropriate.

## 6.1 Page

```python
class Page(BaseModel):
    paper_id: str
    filename: str
    page_number: int
    text: str
```

`page_number` should be 1-indexed for display to users.

---

## 6.2 Chunk

```python
class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    filename: str
    page_number: int
    text: str
```

Optional fields:

```python
start_char: int | None = None
end_char: int | None = None
```

---

## 6.3 Claim

```python
class Claim(BaseModel):
    claim_id: str
    text: str
```

Claims should be atomic and independently verifiable.

---

## 6.4 RetrievedEvidence

```python
class RetrievedEvidence(BaseModel):
    chunk_id: str
    paper_id: str
    filename: str
    page_number: int
    text: str
    score: float | None = None
```

---

## 6.5 VerificationResult

```python
class VerificationResult(BaseModel):
    claim_id: str
    claim_text: str
    label: Literal[
        "SUPPORTED",
        "PARTIALLY_SUPPORTED",
        "UNSUPPORTED",
    ]
    reason: str
    evidence_ids: list[str]
```

The model must only cite evidence IDs that were actually provided to it.

---

# 7. Module 1 — PDF Ingestion

File:

```text
src/pdf_parser.py
```

Implement:

```python
def extract_pdf_pages(
    pdf_path: str,
    paper_id: str | None = None,
) -> list[Page]:
    ...
```

Requirements:

- use PyMuPDF
- extract page-by-page
- preserve page number
- skip empty pages
- normalize excessive whitespace
- keep original filename
- create a stable `paper_id`
- raise a clear exception for unreadable PDFs

For the MVP, assume PDFs contain selectable text.

Do not implement OCR.

## Acceptance criteria

Given a normal academic PDF:

```python
pages = extract_pdf_pages("paper.pdf")
```

should return:

```text
[
  Page(page_number=1, ...),
  Page(page_number=2, ...),
  ...
]
```

and page metadata must remain correct.

---

# 8. Module 2 — Chunking

File:

```text
src/chunker.py
```

Implement a simple deterministic text chunker.

Suggested configuration:

```python
CHUNK_SIZE_WORDS = 350
CHUNK_OVERLAP_WORDS = 50
```

For the first version, chunk **within individual pages**.

Do not create chunks that span multiple pages.

This makes citation display reliable.

Implement:

```python
def chunk_pages(
    pages: list[Page],
    chunk_size_words: int = 350,
    overlap_words: int = 50,
) -> list[Chunk]:
    ...
```

Chunk ID format:

```text
{paper_id}_p{page_number}_c{chunk_index}
```

Example:

```text
smith2024_p7_c2
```

## Important tradeoff

Page-local chunking is less sophisticated than semantic chunking, but it makes page-level citations dependable.

For the one-day MVP, citation correctness is more important than advanced chunking.

## Acceptance criteria

- no empty chunks
- every chunk has a filename
- every chunk has a page number
- every chunk has a unique chunk ID
- overlap works correctly
- chunks never cross page boundaries

---

# 9. Module 3 — Embeddings

File:

```text
src/embeddings.py
```

Create a thin wrapper around the embedding API.

Implement:

```python
def embed_texts(texts: list[str]) -> list[list[float]]:
    ...
```

and optionally:

```python
def embed_query(text: str) -> list[float]:
    ...
```

Requirements:

- batch calls where possible
- validate empty input
- keep model name configurable
- do not scatter API calls throughout the codebase

Put model configuration in:

```text
src/config.py
```

Example:

```python
EMBEDDING_MODEL = "..."
VERIFICATION_MODEL = "..."
```

Do not make model names part of business logic.

---

# 10. Module 4 — Vector Store

File:

```text
src/vector_store.py
```

Use ChromaDB.

Implement a wrapper class:

```python
class PaperVectorStore:

    def add_chunks(self, chunks: list[Chunk]) -> None:
        ...

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedEvidence]:
        ...

    def reset(self) -> None:
        ...
```

Each stored record must include metadata:

```json
{
  "chunk_id": "...",
  "paper_id": "...",
  "filename": "...",
  "page_number": 7
}
```

The raw chunk text must also be stored.

## Acceptance criteria

If a chunk contains:

```text
Retrieval augmentation reduced unsupported factual generations.
```

then a query such as:

```text
RAG reduces hallucination
```

should retrieve that passage near the top.

---

# 11. Module 5 — Atomic Claim Extraction

File:

```text
src/claim_extractor.py
```

Input:

```text
A paragraph, summary, literature review, or related-work passage.
```

Output:

```python
list[Claim]
```

Implement:

```python
def extract_claims(text: str) -> list[Claim]:
    ...
```

Use structured LLM output.

## Claim extraction rules

A claim should:

- express one independently verifiable proposition
- preserve the meaning of the original text
- avoid combining multiple unrelated assertions
- retain important qualifiers
- retain numerical values
- retain named methods/datasets/domains
- not invent information
- not convert opinions into factual claims

Example input:

```text
RAG reduces hallucination and outperforms fine-tuning
on biomedical and legal QA benchmarks.
```

Desired output:

```text
1. RAG reduces hallucination.
2. RAG outperforms fine-tuning on biomedical QA benchmarks.
3. RAG outperforms fine-tuning on legal QA benchmarks.
```

## Prompt

Create:

```text
prompts/claim_extraction.txt
```

Suggested content:

```text
You are extracting atomic factual claims from academic writing.

A claim must be independently verifiable against research evidence.

Rules:
- Split compound statements into separate claims.
- Preserve important qualifiers, entities, domains, and numerical values.
- Do not add facts that are not present in the input.
- Do not rewrite vague opinions as factual claims.
- Keep claims concise but semantically faithful.

Return structured output only.
```

## Acceptance criteria

A compound sentence containing three independently testable statements should normally produce three claims.

---

# 12. Module 6 — Retrieval

File:

```text
src/retriever.py
```

Implement:

```python
def retrieve_evidence(
    claim: Claim,
    vector_store: PaperVectorStore,
    top_k: int = 5,
) -> list[RetrievedEvidence]:
    ...
```

Default:

```python
top_k = 5
```

No query rewriting in the MVP.

No reranking in the MVP.

No multi-hop retrieval in the MVP.

Keep the retrieval stage observable so later experiments can compare different strategies.

---

# 13. Module 7 — Claim Verification

File:

```text
src/verifier.py
```

Implement:

```python
def verify_claim(
    claim: Claim,
    evidence: list[RetrievedEvidence],
) -> VerificationResult:
    ...
```

Use structured output.

The verifier must choose exactly one label:

```text
SUPPORTED
PARTIALLY_SUPPORTED
UNSUPPORTED
```

## Label definitions

### SUPPORTED

Use when the supplied evidence clearly supports the essential meaning of the claim.

The exact wording does not need to match.

### PARTIALLY_SUPPORTED

Use when:

- only part of the claim is supported
- a qualifier is unsupported
- the evidence supports a narrower statement
- a numerical or scope claim is not fully justified
- the evidence supports the direction but not the full strength of the claim

### UNSUPPORTED

Use when:

- the retrieved evidence does not provide sufficient support
- the evidence is irrelevant
- the claim contains a key assertion absent from the evidence

Important:

`UNSUPPORTED` does not mean globally false.

---

# 14. Verification Prompt

Create:

```text
prompts/claim_verification.txt
```

Suggested prompt:

```text
You are an evidence-grounded scientific claim verifier.

Your task is to determine whether the supplied evidence supports the claim.

You must reason only from the provided evidence.

Labels:

SUPPORTED:
The evidence directly supports the essential meaning of the claim.

PARTIALLY_SUPPORTED:
The evidence supports part of the claim, but an important component,
qualifier, scope condition, comparison, numerical value, or generalization
is not sufficiently supported.

UNSUPPORTED:
The supplied evidence does not sufficiently support the claim.

Important rules:

1. Do not use outside knowledge.
2. Do not assume that an unsupported claim is false.
3. Be strict about:
   - numerical values
   - causal claims
   - comparisons
   - claims of statistical significance
   - words such as "always", "consistently", "all", "best", or "state-of-the-art"
4. Only cite evidence IDs that were provided.
5. Keep the explanation concise.
6. Return structured output only.
```

Construct the runtime input like:

```text
CLAIM:
{claim_text}

EVIDENCE:

[E1]
Source: paper_a.pdf
Page: 4
Text:
...

[E2]
Source: paper_b.pdf
Page: 9
Text:
...
```

---

# 15. Module 8 — End-to-End Pipeline

File:

```text
src/pipeline.py
```

Create a high-level orchestration layer.

Suggested interface:

```python
class ClaimVerificationPipeline:

    def ingest_pdfs(
        self,
        pdf_paths: list[str],
    ) -> None:
        ...

    def verify_text(
        self,
        text: str,
    ) -> list[VerificationResult]:
        ...
```

Conceptual flow:

```python
claims = extract_claims(text)

results = []

for claim in claims:
    evidence = retrieve_evidence(
        claim,
        vector_store,
        top_k=5,
    )

    result = verify_claim(
        claim,
        evidence,
    )

    results.append(result)

return results
```

Do not introduce agent loops.

---

# 16. Streamlit UI

File:

```text
app.py
```

Keep the UI simple.

## Section A — Paper Upload

Allow multiple PDF uploads.

Display:

```text
3 papers uploaded
128 pages extracted
412 chunks indexed
```

Provide a button:

```text
Build / Rebuild Index
```

---

## Section B — Text to Verify

Large text area:

```text
Paste an AI-generated literature review,
paper summary, or research paragraph here.
```

Button:

```text
Verify Claims
```

---

## Section C — Results

For each claim display:

```text
Claim 1

RAG significantly reduces hallucination in biomedical QA.

Verdict: PARTIALLY_SUPPORTED

Reason:
The evidence reports lower unsupported generation,
but does not establish statistical significance.

Evidence:

paper_a.pdf — Page 6
"Retrieval augmentation reduced unsupported factual generations..."

paper_b.pdf — Page 4
"..."
```

Suggested visual treatment:

```text
SUPPORTED            -> green
PARTIALLY_SUPPORTED  -> yellow/orange
UNSUPPORTED          -> red
```

Do not depend on color alone; always display the text label.

---

# 17. Groundedness Summary

At the top of results, show:

```text
Total claims: 8
Supported: 5
Partially supported: 2
Unsupported: 1
```

Optional simple score:

```python
SUPPORTED = 1.0
PARTIALLY_SUPPORTED = 0.5
UNSUPPORTED = 0.0
```

Then:

```text
Groundedness score =
sum(claim_scores) / number_of_claims
```

Display:

```text
Groundedness: 75%
```

Important:

Document clearly in the UI and README that this score is a heuristic summary, not a scientifically validated universal hallucination metric.

---

# 18. Evaluation Dataset

Create:

```text
data/eval/sample_eval.json
```

Schema:

```json
[
  {
    "id": "claim_001",
    "claim": "RAG reduces unsupported factual generation.",
    "label": "SUPPORTED",
    "expected_evidence": [
      {
        "filename": "paper_a.pdf",
        "page_number": 6
      }
    ]
  }
]
```

Target for the first day:

```text
15-30 manually labeled claims
```

Aim for rough balance across:

```text
SUPPORTED
PARTIALLY_SUPPORTED
UNSUPPORTED
```

The evaluation dataset may initially use only a few selected papers.

---

# 19. Evaluation Metrics

File:

```text
src/evaluation.py
```

Implement:

```python
def classification_accuracy(...):
    ...

def confusion_matrix(...):
    ...

def retrieval_hit_at_k(...):
    ...
```

Minimum metrics:

## 19.1 Claim Classification Accuracy

```text
correct predictions / total claims
```

---

## 19.2 Per-Class Counts

Show:

```text
SUPPORTED:
correct / total

PARTIALLY_SUPPORTED:
correct / total

UNSUPPORTED:
correct / total
```

If easy, also compute precision, recall, and F1.

---

## 19.3 Retrieval Hit@K

If the expected evidence page is known:

```text
Hit@K = whether at least one expected source page
appears in the top-K retrieved chunks
```

Report:

```text
Hit@1
Hit@3
Hit@5
```

This is important because it helps distinguish:

```text
retrieval failure
```

from:

```text
verification failure
```

---

# 20. Error Analysis

For every incorrect evaluation example, save enough information to inspect:

```text
claim
ground-truth label
predicted label
retrieved evidence
expected evidence
verifier explanation
```

Optional output:

```text
data/eval/error_analysis.json
```

This will be useful for research discussion.

---

# 21. Tests

Use pytest.

The MVP does not need perfect test coverage, but critical deterministic logic should be tested.

## test_pdf_parser.py

Test:

- valid PDF
- page numbers
- empty page handling

## test_chunker.py

Test:

- unique chunk IDs
- correct page metadata
- overlap behavior
- no cross-page chunks
- no empty chunks

## test_claim_extractor.py

Mock the model call if practical.

Test schema parsing.

## test_retriever.py

Use a tiny deterministic fixture.

Verify retrieval returns metadata.

## test_verifier.py

Mock structured output.

Verify:

- valid labels only
- evidence IDs are parsed
- no unknown label passes validation

## test_evaluation.py

Test:

- accuracy
- confusion matrix
- Hit@K

---

# 22. Logging and Debuggability

The project should make it easy to understand failures.

At minimum log:

```text
number of pages extracted
number of chunks created
number of claims extracted
retrieved chunk IDs per claim
retrieval scores
verification label
```

Do not log API keys.

Use Python logging rather than scattered `print()` statements where practical.

---

# 23. Configuration

File:

```text
src/config.py
```

Centralize:

```python
CHUNK_SIZE_WORDS = 350
CHUNK_OVERLAP_WORDS = 50
TOP_K = 5

EMBEDDING_MODEL = "..."
CLAIM_EXTRACTION_MODEL = "..."
VERIFICATION_MODEL = "..."
```

Allow environment-variable overrides where reasonable.

---

# 24. Requirements File

Create `requirements.txt` with the minimal necessary packages.

Expected dependencies include approximately:

```text
streamlit
pymupdf
openai
chromadb
pydantic
python-dotenv
pytest
pandas
```

Avoid unnecessary frameworks.

---

# 25. README

The README should include:

1. What the project does
2. Why it exists
3. Architecture diagram
4. Setup instructions
5. How to add `OPENAI_API_KEY`
6. How to run:

```bash
streamlit run app.py
```

7. Example workflow
8. Label definitions
9. Evaluation instructions
10. Current limitations
11. Future work

Include this limitation prominently:

> The system evaluates whether claims are supported by the uploaded evidence. It does not determine universal factual truth.

---

# 26. One-Day Implementation Schedule

## Phase 1 — 09:00-10:00
### Repository setup

Tasks:

- create project structure
- create virtual environment instructions
- install dependencies
- add `.env.example`
- add config
- create Pydantic schemas
- create basic Streamlit shell

Deliverable:

```text
Application launches successfully.
```

---

## Phase 2 — 10:00-11:30
### PDF parsing and chunking

Tasks:

- implement `extract_pdf_pages`
- implement `chunk_pages`
- preserve source metadata
- add tests

Deliverable:

```text
PDF → list of citation-ready chunks
```

---

## Phase 3 — 11:30-12:30
### Embeddings and vector retrieval

Tasks:

- implement embedding wrapper
- implement Chroma store
- ingest chunks
- implement semantic search

Manual test:

Query:

```text
RAG reduces hallucination
```

should retrieve semantically related passages.

Deliverable:

```text
claim-like query → relevant evidence chunks
```

---

## Phase 4 — 13:30-14:30
### Atomic claim extraction

Tasks:

- create claim extraction prompt
- enforce structured output
- add UI preview of extracted claims

Deliverable:

```text
paragraph → atomic claims
```

---

## Phase 5 — 14:30-16:00
### Claim verification

Tasks:

- implement retrieval per claim
- implement verification prompt
- enforce structured output
- connect evidence IDs
- return label + reason

Deliverable:

```text
claim + evidence → verification result
```

---

## Phase 6 — 16:00-17:00
### UI integration

Tasks:

- show uploaded papers
- show index statistics
- paste text
- verify claims
- render result cards
- show filename/page/evidence

Deliverable:

```text
end-to-end usable demo
```

---

## Phase 7 — 17:00-18:00
### Evaluation

Tasks:

- manually create 15-30 labeled claims
- implement accuracy
- implement retrieval Hit@K
- generate evaluation summary

Deliverable:

```text
basic quantitative evaluation
```

---

## Phase 8 — 18:00-19:00
### Cleanup

Tasks:

- fix obvious bugs
- improve README
- remove dead code
- confirm fresh install works
- run tests
- demo complete workflow

Deliverable:

```text
clean MVP suitable for a research demo
```

---

# 27. Definition of Done

The MVP is done when all of the following work:

- [ ] User can upload at least 3 PDFs.
- [ ] PDF text is extracted page-by-page.
- [ ] Chunks preserve page-level metadata.
- [ ] Chunks are embedded and indexed.
- [ ] User can paste a paragraph.
- [ ] Paragraph is decomposed into atomic claims.
- [ ] Top-5 evidence is retrieved for every claim.
- [ ] Every claim receives exactly one valid verdict.
- [ ] Every verdict has a concise explanation.
- [ ] Evidence citations include filename and page.
- [ ] UI renders the full end-to-end workflow.
- [ ] A small human-labeled evaluation dataset exists.
- [ ] Classification accuracy can be computed.
- [ ] Retrieval Hit@K can be computed.
- [ ] Tests for deterministic modules pass.
- [ ] README explains setup, usage, and limitations.
- [ ] No Agent / ReAct / multi-agent framework is required.

---

# 28. Quality Priorities

If time becomes limited, prioritize in this order:

1. Correct page/source metadata
2. End-to-end functionality
3. Atomic claim quality
4. Retrieval quality
5. Verification reliability
6. Evaluation
7. UI polish

Do **not** sacrifice source traceability for visual polish.

---

# 29. Known Failure Modes

Expect the following.

## 29.1 Retrieval failure

The correct evidence exists but is not in top-K.

Potential future fixes:

- hybrid search
- query expansion
- reranking
- larger K
- semantic chunking

---

## 29.2 Overly broad claims

Example:

```text
RAG always improves factuality across all scientific domains.
```

Evidence may only support a narrow domain.

The verifier should be strict about generalization.

---

## 29.3 Numerical hallucinations

Example:

```text
The method improves accuracy by 17%.
```

Evidence only reports a general improvement.

This should not be `SUPPORTED`.

---

## 29.4 Citation mismatch

A semantically similar chunk may not actually prove the claim.

The LLM verifier must evaluate entailment, not only similarity.

---

## 29.5 PDF parsing issues

Equations, columns, footnotes, and tables may extract imperfectly.

Document this limitation rather than implementing a complex parser on day one.

---

# 30. Future Research Extensions

Only implement these after the MVP works.

Potential experiments:

## Retrieval experiments

Compare:

```text
Top-K = 1, 3, 5, 10
```

Measure:

```text
Hit@K
verification accuracy
```

---

## Chunking experiments

Compare:

```text
small chunks
large chunks
page-based chunks
semantic chunks
```

---

## Retrieval strategy experiments

Compare:

```text
dense retrieval
BM25
hybrid retrieval
dense + reranker
```

---

## Verification model experiments

Compare multiple LLMs or NLI models using the same retrieved evidence.

Measure:

```text
accuracy
precision
recall
F1
cost
latency
```

---

## Citation correctness

Evaluate whether cited evidence truly supports the output.

Possible metrics:

```text
citation precision
citation recall
citation completeness
```

---

## Add CONTRADICTED

Later extend labels to:

```text
SUPPORTED
PARTIALLY_SUPPORTED
UNSUPPORTED
CONTRADICTED
```

Do this only after the original three-class evaluator is stable.

---

# 31. Optional Second-Day Agent Extension

Do not build this on day one.

A future agentic version could expose tools such as:

```python
search_papers(query)
read_page(paper_id, page_number)
retrieve_more_evidence(claim)
compare_evidence(claim, evidence)
```

An agent could decide when the first retrieval is insufficient.

Example:

```text
Claim
 ↓
Search
 ↓
Evidence insufficient?
 ├─ No → Verify
 └─ Yes
      ↓
 Reformulate query
      ↓
 Search again
```

This could later be implemented as an agentic retrieval experiment.

However, the deterministic baseline must remain available for comparison.

---

# 32. Codex Working Instructions

When implementing this plan:

1. Work incrementally.
2. Keep the application runnable after each major phase.
3. Prefer simple Python functions over abstractions that are not yet necessary.
4. Do not introduce Agent frameworks.
5. Do not introduce LangChain or LlamaIndex.
6. Use structured model outputs rather than parsing free-form prose.
7. Preserve filename and page metadata through the entire pipeline.
8. Add tests while implementing deterministic modules.
9. If a design detail is unspecified, choose the simplest reasonable implementation.
10. Record significant deviations from this plan in the README.
11. Do not silently broaden project scope.
12. Do not add cloud infrastructure unless required.
13. Keep API-specific code isolated behind small wrappers.
14. Avoid premature optimization.
15. Before declaring completion:
    - run tests
    - run the Streamlit app
    - verify one complete example manually
    - check that every displayed citation maps to an actual retrieved chunk

---

# 33. Suggested Initial Codex Task Sequence

Codex should execute roughly in this order:

```text
Task 1
Create repository structure, config, schemas, and Streamlit shell.

Task 2
Implement PDF parser and tests.

Task 3
Implement page-local chunking and tests.

Task 4
Implement embedding wrapper.

Task 5
Implement Chroma vector store.

Task 6
Add manual semantic retrieval smoke test.

Task 7
Implement structured atomic claim extraction.

Task 8
Implement evidence retrieval.

Task 9
Implement structured claim verification.

Task 10
Implement end-to-end pipeline.

Task 11
Connect pipeline to Streamlit.

Task 12
Create sample evaluation dataset.

Task 13
Implement classification and retrieval metrics.

Task 14
Run tests and fix failures.

Task 15
Polish README and demo flow.
```

---

# 34. Final Product Mental Model

The MVP should remain conceptually simple:

```text
              PAPERS
                ↓
        parse → chunk → embed
                ↓
          VECTOR DATABASE
                ↑
                │
AI TEXT → atomic claims
                ↓
          retrieve evidence
                ↓
          verify grounding
                ↓
  SUPPORTED / PARTIAL / UNSUPPORTED
                ↓
       evidence + citations
```

The main research value is not that the system uses an LLM.

The research value comes from making the verification process:

- evidence-grounded
- inspectable
- measurable
- reproducible
- decomposable into retrieval and verification errors

That should remain the guiding principle for the entire implementation.
