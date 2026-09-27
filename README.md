# Academic Claim Verifier

Academic Claim Verifier checks whether claims in a piece of academic writing are supported by a set of uploaded papers. It is meant for literature reviews, related-work drafts, paper summaries, and research notes where every conclusion should be traceable to a source passage.

The tool does not decide whether a claim is universally true. It only asks whether the uploaded evidence supports it. In particular, **`UNSUPPORTED` does not mean `FALSE`**.

## How it works

```text
PDFs → page text → page-local chunks → embeddings → Chroma index
                                                    ↑
draft text → atomic claims → top-K retrieval → verification
                                                    ↓
                              verdict, explanation, and source pages
```

Each PDF is processed page by page. Chunks never cross page boundaries, so every retrieved passage retains a dependable filename and page number. The verification step receives only those retrieved passages and must return one of three labels:

- `SUPPORTED` — the evidence supports the essential meaning of the claim.
- `PARTIALLY_SUPPORTED` — the evidence supports a narrower or incomplete version of the claim.
- `UNSUPPORTED` — the uploaded evidence does not provide enough support.

The verifier is deliberately strict about numbers, percentages, comparisons, causal language, statistical significance, and broad terms such as “all,” “always,” and “state-of-the-art.”

## Example

Suppose an uploaded paper contains this sentence on page 6:

> Retrieval augmentation reduced unsupported factual generations in the biomedical question-answering experiment.

Now consider this draft sentence:

> Retrieval augmentation reduced unsupported factual generations by 50% across all scientific domains.

The claim extractor can separate it into independently checkable statements:

1. Retrieval augmentation reduced unsupported factual generations.
2. The reduction was 50%.
3. The reduction occurred across all scientific domains.

Given only the passage above, the first claim may be `SUPPORTED`, while the numerical and cross-domain claims should be `UNSUPPORTED`. The result includes the passage, `paper.pdf — Page 6`, and its retrieval score, so the decision can be inspected instead of taken on trust.

This is an illustrative workflow, not a research result or an evaluation example.

## Setup

Requirements:

- Python 3.11 or newer
- an OpenAI API key
- PDFs with selectable text

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and add your key:

```dotenv
OPENAI_API_KEY=your-key-here
```

The model names and Chroma location can also be changed in `.env`. The defaults are documented in [.env.example](.env.example).

## Run the app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

In the app:

1. Upload one or more PDFs.
2. Click **Build / Rebuild Index**.
3. Paste the text you want to check.
4. Click **Verify Claims**.
5. Review each verdict alongside its retrieved passages and page references.

The app only calls the API after an explicit button click. Parsed papers, the pipeline, and results are kept in Streamlit session state to avoid unnecessary work during reruns.

## Groundedness summary

The interface reports a simple aggregate score:

```text
SUPPORTED            = 1.0
PARTIALLY_SUPPORTED  = 0.5
UNSUPPORTED          = 0.0
```

The average is displayed as a percentage. It is a convenience for scanning a batch of claims, not a validated hallucination metric.

## Tests

The automated suite uses deterministic embeddings and mocked structured model responses, so it does not need an API key:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

To run the live embedding and retrieval smoke test:

```powershell
.\.venv\Scripts\python.exe scripts\semantic_retrieval_smoke.py
```

## Evaluation

Evaluation keeps retrieval and verification separate:

- classification accuracy, confusion matrix, and per-class precision/recall/F1 measure the final verdicts;
- Hit@1, Hit@3, and Hit@5 measure whether the expected source page was retrieved.

That distinction matters: a wrong verdict may come from a retrieval miss or from incorrect interpretation of good evidence.

The included `data/eval/sample_eval.json` is only a schema/demo fixture. Its entries are marked as placeholders because no manually reviewed paper set is included. To run a real evaluation, add reviewed PDFs to `data/papers/`, replace the placeholders with manually labeled claims and page references, set their status to `MANUALLY_REVIEWED`, and run:

```powershell
.\.venv\Scripts\python.exe scripts\run_evaluation.py
```

The runner refuses to report metrics when the key, papers, reviewed labels, or expected pages are missing.

## Project layout

```text
app.py                    Streamlit interface
src/pdf_parser.py         page-level PDF extraction
src/chunker.py            page-local chunking
src/embeddings.py         embedding API wrapper
src/vector_store.py       persistent Chroma index
src/claim_extractor.py    structured claim extraction
src/retriever.py          top-K evidence retrieval
src/verifier.py           evidence-grounded verification
src/pipeline.py           end-to-end orchestration
src/evaluation.py         metrics and error analysis
prompts/                  extraction and verification prompts
scripts/                  smoke test and evaluation runner
tests/                    API-key-free test suite
```

## Limitations

- Scanned PDFs are not supported; there is no OCR.
- Multi-column layouts, equations, footnotes, and tables may extract imperfectly.
- Page-local chunks improve citation reliability but may split semantic units.
- Dense retrieval can miss relevant evidence or return similar but non-entailing passages.
- Verification quality depends on the selected model and the retrieved context.
- The MVP has no contradiction label and performs no web search.

Useful next steps include a manually reviewed evaluation set, chunk-size and top-K experiments, and citation-quality measurement. Hybrid retrieval and reranking are intentionally left outside the baseline.

## Acknowledgements

This project was completed with assistance from OpenAI Codex, including implementation review and test development.
