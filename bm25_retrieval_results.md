# Retrieval Experiment Comparison: Whole Document, Blind Chunking, V1–V3, and BM25

This experiment compares the frozen 10-query benchmark across:

- Whole-document dense retrieval
- Blind fixed-size chunking
- Structure-aware V1
- Structure-aware V2
- Structure-aware V3
- BM25 over the same V3 `retrieval_text`

The goal was to understand how the different retrieval strategies fail, not to force a single retriever to solve every query.

## Preferred-Source Rank Comparison

| Query | Whole Doc | Blind | V1 | V2 | V3 | BM25 |
|---|---:|---:|---:|---:|---:|---:|
| T01 | 2 | 2 | 2 | 1 | 1 | 2* |
| T02 | 2 | 2 | 3 | 3 | 3 | **1** |
| T03 | **1** | **1** | **1** | **1** | **1** | **1** |
| T04 | 2 | **1** | **1** | **1** | **1** | **1** |
| T05 | **1** | **1** | **1** | **1** | **1** | **1** |
| T06 — preferred document | 2 | 7 | 2 | 2 | 2 | 4 |
| T06 — correct §12.5 clause | — | 7 | 4 | 4 | 4 | 4 |
| T07 | 2 | 3 | 5 | 4 | 4 | >5 |
| T08 | **1** | **1** | 2 | 2 | — | 2 |
| T09 | **1** | **1** | **1** | **1** | **1** | >5 |
| T10 | 2 | 5 | 2 | 2 | 2 | >5 |

`*` For T01, BM25 first reaches the Employment Agreement at rank 2, but that returned clause is not the answer-bearing non-compete clause.

`—` means the result was not available or not recorded for that experiment.

## What Changed

### Whole-document retrieval
Whole documents preserved broad semantic context well, but evidence localization was weak.

### Blind chunking
Blind chunks improved locality, but splitting legal cross-references hurt queries such as T06 and T10.

### Structure-aware V1
V1 preserved legal structure and graph dependencies, but embedding all resolved dependencies created overly broad retrieval representations.

### Structure-aware V2
V2 reduced contamination by embedding only reachable definitions. It improved some rankings but could still include more definitions than necessary.

### Structure-aware V3
V3 used bounded, actual defined-term chains instead of the full dependency closure. It produced the cleanest dense representation, but preferred-source rankings remained mostly similar to V2. This suggested that the remaining failures were no longer primarily a chunk-representation problem.

### BM25
BM25 produced a different failure pattern.

It improved exact lexical queries:

- **T02:** preferred Legal Analysis improved from rank 3 to rank 1.
- **T03, T04, T05:** remained rank 1.

But it was much weaker on semantic/paraphrase-heavy or authority-sensitive queries:

- **T07:** preferred source fell outside Top 5.
- **T09:** Settlement fell outside Top 5.
- **T10:** Settlement fell outside Top 5.
- **T06:** the correct §12.5 clause remained rank 4.

BM25 was strong when rare or exact terms directly identified the relevant text, but it could not reliably understand semantic equivalence such as:

- `more defensible` ≈ `stronger enforcement argument`
- `allowed to work` ≈ `may remain employed`

It also did not understand source authority, chronology, or whether an earlier legal analysis was less authoritative than the final settlement.

## Overall Conclusion

Dense V3 and BM25 make different mistakes.

Dense V3 is stronger at:

- semantic meaning
- paraphrases
- concept similarity

BM25 is stronger at:

- exact terminology
- rare names
- lexical matches
- section-specific wording

Because the two retrievers are complementary, the next experiment should combine them rather than continue tuning either one independently.

```text
Dense V3 Top-K
        +
BM25 Top-K
        ↓
Reciprocal Rank Fusion (RRF)
        ↓
Hybrid candidate ranking
```

The next objective is to make sure the correct evidence reliably survives into a small hybrid candidate set for later reranking and authority-aware selection.
