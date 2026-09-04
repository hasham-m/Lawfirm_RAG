# Whole-Document Retrieval vs Blind Chunk Retrieval

## Experiment Overview

This report compares two retrieval strategies on the same 10-query legal RAG benchmark for the Hamilton & Cole LLP / Carter–Northstar matter (`HC-2025-0142`):

1. **Whole-document retrieval**
2. **Blind fixed-size chunk retrieval**

The goal is to understand how changing the retrieval unit from a full document to fixed-size chunks affects:

- preferred-source ranking,
- retrieval stability,
- source sufficiency,
- semantic relevance,
- legal authority / provenance.

The benchmark queries and preferred documents were kept fixed between experiments.

---

## Retrieval Configurations

### Whole-Document Retrieval

Each canonical legal document was embedded as one retrieval unit.

For each query:

```text
query
→ query embedding
→ cosine similarity against all document embeddings
→ ranked documents
```

The whole-document experiment was evaluated primarily using Top-1 and Top-2 retrieval.

### Blind Chunk Retrieval

Documents were split using a fixed-size sliding window:

```text
chunk size = 300 words
overlap    = 50 words
step       = 250 words
```

Chunk boundaries did **not** use:

- headings,
- clauses,
- sections,
- paragraphs,
- document structure.

Each chunk preserved parent-document provenance metadata such as:

- `document_id`
- `title`
- `document_type`
- page range
- chunk index

For each query:

```text
query
→ query embedding
→ cosine similarity against all chunk embeddings
→ ranked chunks
```

The blind-chunk benchmark reports Top-5 results and preferred-document Recall@K.

---

# Aggregate Results

## Whole-Document Retrieval

| Metric | Result |
|---|---:|
| Preferred-document Recall@1 | **40%** |
| Preferred-document Recall@2 | **100%** |
| MRR | **0.7000** |

## Blind Chunk Retrieval

| Metric | Result |
|---|---:|
| Preferred-document Recall@1 | **50%** |
| Preferred-document Recall@2 | **70%** |
| Preferred-document Recall@3 | **80%** |
| Preferred-document Recall@5 | **90%** |
| MRR | **0.6676** |

## Initial Interpretation

Blind chunking improved **first-rank precision** on some locally specific questions, but whole-document retrieval was substantially more stable for recovering the benchmark's preferred authoritative source.

Most importantly:

```text
Whole-document Recall@2 = 100%
Blind-chunk Recall@2     = 70%
Blind-chunk Recall@3     = 80%
Blind-chunk Recall@5     = 90%
```

The whole-document baseline therefore recovered the preferred document within the first two results for **all 10 benchmark queries**.

---

# Query-by-Query Comparison

## T01

**Query**

> How long and how far did Carter's original non-compete restrict him?

**Preferred document:** `DOC-2025-0142-02` — Employment Agreement

### Whole-Document Retrieval

| Rank | Document |
|---|---|
| 1 | `DOC-2025-0142-07` — Carter Opposition |
| 2 | `DOC-2025-0142-02` — Employment Agreement |

**Preferred-document rank:** **2**

### Blind Chunk Retrieval

| Rank | Chunk | Parent Document | Score |
|---|---|---|---:|
| 1 | `DOC-2025-0142-07::blind::0000` | `DOC-2025-0142-07` | 0.720592 |
| 2 | `DOC-2025-0142-02::blind::0008` | `DOC-2025-0142-02` | 0.715545 |
| 3 | `DOC-2025-0142-01::blind::0000` | `DOC-2025-0142-01` | 0.705258 |
| 4 | `DOC-2025-0142-10::blind::0000` | `DOC-2025-0142-10` | 0.701885 |
| 5 | `DOC-2025-0142-09::blind::0000` | `DOC-2025-0142-09` | 0.700881 |

**Preferred-document rank:** **2**

### Comparison

**Tie.**

Both systems rank the Opposition first and the Employment Agreement second.

The Opposition is highly semantically relevant because it discusses the fifty-mile restriction, but the Employment Agreement is the preferred operative source for the original restriction.

---

## T02

**Query**

> Why was Carter's fifty-mile non-compete potentially difficult to enforce?

**Preferred document:** `DOC-2025-0142-04` — Restrictive Covenant Preliminary Legal Analysis

### Whole-Document Retrieval

| Rank | Document |
|---|---|
| 1 | `DOC-2025-0142-07` — Carter Opposition |
| 2 | `DOC-2025-0142-04` — Research / Legal Analysis Memo |

**Preferred-document rank:** **2**

### Blind Chunk Retrieval

| Rank | Chunk | Parent Document | Score |
|---|---|---|---:|
| 1 | `DOC-2025-0142-07::blind::0000` | `DOC-2025-0142-07` | 0.728825 |
| 2 | `DOC-2025-0142-04::blind::0000` | `DOC-2025-0142-04` | 0.722497 |
| 3 | `DOC-2025-0142-10::blind::0000` | `DOC-2025-0142-10` | 0.715198 |
| 4 | `DOC-2025-0142-01::blind::0000` | `DOC-2025-0142-01` | 0.709133 |
| 5 | `DOC-2025-0142-02::blind::0008` | `DOC-2025-0142-02` | 0.690373 |

**Preferred-document rank:** **2**

### Comparison

**Tie.**

Both systems prefer the Opposition semantically, while the benchmark prefers the internal legal analysis because it provides the more appropriate source role for explaining enforceability concerns.

---

## T03

**Query**

> What did Judge Maren think about the fifty-mile geographic restriction?

**Preferred document:** `DOC-2025-0142-08` — Preliminary Injunction Hearing Notes

### Whole-Document Retrieval

| Rank | Document |
|---|---|
| 1 | `DOC-2025-0142-08` — Hearing Notes |
| 2 | `DOC-2025-0142-07` — Carter Opposition |

**Preferred-document rank:** **1**

### Blind Chunk Retrieval

| Rank | Chunk | Parent Document | Score |
|---|---|---|---:|
| 1 | `DOC-2025-0142-08::blind::0000` | `DOC-2025-0142-08` | 0.688132 |
| 2 | `DOC-2025-0142-07::blind::0000` | `DOC-2025-0142-07` | 0.658171 |
| 3 | `DOC-2025-0142-10::blind::0000` | `DOC-2025-0142-10` | 0.653892 |
| 4 | `DOC-2025-0142-04::blind::0000` | `DOC-2025-0142-04` | 0.651025 |
| 5 | `DOC-2025-0142-02::blind::0008` | `DOC-2025-0142-02` | 0.641406 |

**Preferred-document rank:** **1**

### Comparison

**Tie.**

The query contains a highly distinctive source/context signal — Judge Maren's view — so both retrieval strategies correctly rank the Hearing Notes first.

---

## T04

**Query**

> Under the original employment agreement, did simply accepting employment with a competitor automatically violate the agreement?

**Preferred document:** `DOC-2025-0142-02` — Employment Agreement

### Whole-Document Retrieval

| Rank | Document |
|---|---|
| 1 | `DOC-2025-0142-01` — Client Intake Memorandum |
| 2 | `DOC-2025-0142-02` — Employment Agreement |

**Preferred-document rank:** **2**

### Blind Chunk Retrieval

| Rank | Chunk | Parent Document | Score |
|---|---|---|---:|
| 1 | `DOC-2025-0142-02::blind::0008` | `DOC-2025-0142-02` | 0.722736 |
| 2 | `DOC-2025-0142-02::blind::0009` | `DOC-2025-0142-02` | 0.721775 |
| 3 | `DOC-2025-0142-02::blind::0007` | `DOC-2025-0142-02` | 0.699767 |
| 4 | `DOC-2025-0142-02::blind::0011` | `DOC-2025-0142-02` | 0.692182 |
| 5 | `DOC-2025-0142-01::blind::0000` | `DOC-2025-0142-01` | 0.690590 |

**Preferred-document rank:** **1**

### Comparison

**Blind chunking wins.**

The whole Employment Agreement contains many unrelated provisions, which dilute the embedding.

Blind chunking isolates the restrictive-covenant region, allowing several Agreement chunks to dominate the ranking.

This is a clear example of chunking improving retrieval when the relevant evidence is locally concentrated.

---

## T05

**Query**

> What percentage of a publicly traded competitor could Carter own without violating the agreement solely because of the investment?

**Preferred document:** `DOC-2025-0142-02` — Employment Agreement

### Whole-Document Retrieval

| Rank | Document |
|---|---|
| 1 | `DOC-2025-0142-02` — Employment Agreement |
| 2 | `DOC-2025-0142-09` — Settlement Agreement |

**Preferred-document rank:** **1**

### Blind Chunk Retrieval

| Rank | Chunk | Parent Document | Score |
|---|---|---|---:|
| 1 | `DOC-2025-0142-02::blind::0007` | `DOC-2025-0142-02` | 0.710808 |
| 2 | `DOC-2025-0142-09::blind::0000` | `DOC-2025-0142-09` | 0.699224 |
| 3 | `DOC-2025-0142-01::blind::0000` | `DOC-2025-0142-01` | 0.691475 |
| 4 | `DOC-2025-0142-07::blind::0000` | `DOC-2025-0142-07` | 0.689212 |
| 5 | `DOC-2025-0142-02::blind::0008` | `DOC-2025-0142-02` | 0.686745 |

**Preferred-document rank:** **1**

### Comparison

**Tie on preferred-source rank.**

Blind chunking may still be more context-efficient because it retrieves a local Agreement chunk rather than requiring the full contract.

---

## T06

**Query**

> How long was Carter prohibited from directly soliciting customers he had been responsible for before leaving Northstar?

**Preferred document:** `DOC-2025-0142-02` — Employment Agreement

### Whole-Document Retrieval

| Rank | Document |
|---|---|
| 1 | `DOC-2025-0142-09` — Settlement Agreement |
| 2 | `DOC-2025-0142-02` — Employment Agreement |

**Preferred-document rank:** **2**

### Blind Chunk Retrieval

| Rank | Chunk | Parent Document | Score |
|---|---|---|---:|
| 1 | `DOC-2025-0142-09::blind::0000` | `DOC-2025-0142-09` | 0.794797 |
| 2 | `DOC-2025-0142-01::blind::0000` | `DOC-2025-0142-01` | 0.781708 |
| 3 | `DOC-2025-0142-07::blind::0000` | `DOC-2025-0142-07` | 0.767057 |
| 4 | `DOC-2025-0142-06::blind::0000` | `DOC-2025-0142-06` | 0.764630 |
| 5 | `DOC-2025-0142-10::blind::0000` | `DOC-2025-0142-10` | 0.764379 |

**Preferred-document rank:** **7**

### Comparison

**Whole-document retrieval wins strongly.**

This is the largest blind-chunk degradation.

Several documents discuss essentially the same six-month customer nonsolicitation issue, so semantically relevant but non-preferred sources outrank the original Agreement.

The Agreement also depends on related contractual concepts such as:

- Restricted Customer,
- Material Responsibility,
- customer nonsolicitation,
- duration,
- separation.

Blind fixed-size boundaries can separate these related concepts, reducing the semantic strength of any single Agreement chunk.

---

## T07

**Query**

> What restriction was considered more defensible than preventing Carter from competing within fifty miles of Westbridge?

**Preferred document:** `DOC-2025-0142-04` — Restrictive Covenant Preliminary Legal Analysis

### Whole-Document Retrieval

| Rank | Document |
|---|---|
| 1 | `DOC-2025-0142-07` — Carter Opposition |
| 2 | `DOC-2025-0142-04` — Legal Analysis Memo |

**Preferred-document rank:** **2**

### Blind Chunk Retrieval

| Rank | Chunk | Parent Document | Score |
|---|---|---|---:|
| 1 | `DOC-2025-0142-07::blind::0000` | `DOC-2025-0142-07` | 0.731155 |
| 2 | `DOC-2025-0142-10::blind::0000` | `DOC-2025-0142-10` | 0.713125 |
| 3 | `DOC-2025-0142-04::blind::0000` | `DOC-2025-0142-04` | 0.710764 |
| 4 | `DOC-2025-0142-02::blind::0008` | `DOC-2025-0142-02` | 0.707508 |
| 5 | `DOC-2025-0142-06::blind::0000` | `DOC-2025-0142-06` | 0.691921 |

**Preferred-document rank:** **3**

### Comparison

**Whole-document retrieval wins slightly.**

The Closing Memo becomes a strong blind-chunk competitor because it summarizes the same outcome using concentrated language.

---

## T08

**Query**

> What did Northstar actually ask the court to stop Carter from doing at Vertex?

**Preferred document:** `DOC-2025-0142-06` — Motion for Preliminary Injunction

### Whole-Document Retrieval

| Rank | Document |
|---|---|
| 1 | `DOC-2025-0142-06` — Motion |
| 2 | `DOC-2025-0142-07` — Opposition |

**Preferred-document rank:** **1**

### Blind Chunk Retrieval

| Rank | Chunk | Parent Document | Score |
|---|---|---|---:|
| 1 | `DOC-2025-0142-06::blind::0000` | `DOC-2025-0142-06` | 0.790543 |
| 2 | `DOC-2025-0142-07::blind::0000` | `DOC-2025-0142-07` | 0.787494 |
| 3 | `DOC-2025-0142-05::blind::0000` | `DOC-2025-0142-05` | 0.751674 |
| 4 | `DOC-2025-0142-09::blind::0000` | `DOC-2025-0142-09` | 0.746614 |
| 5 | `DOC-2025-0142-01::blind::0000` | `DOC-2025-0142-01` | 0.745519 |

**Preferred-document rank:** **1**

### Comparison

**Tie.**

The Motion is both semantically relevant and the correct source role because the question explicitly asks what Northstar requested from the court.

---

## T09

**Query**

> After the dispute was resolved, was Carter still allowed to work for Vertex?

**Preferred document:** `DOC-2025-0142-09` — Settlement Agreement

### Whole-Document Retrieval

| Rank | Document |
|---|---|
| 1 | `DOC-2025-0142-09` — Settlement Agreement |
| 2 | `DOC-2025-0142-10` — Closing Memorandum |

**Preferred-document rank:** **1**

### Blind Chunk Retrieval

| Rank | Chunk | Parent Document | Score |
|---|---|---|---:|
| 1 | `DOC-2025-0142-09::blind::0000` | `DOC-2025-0142-09` | 0.734877 |
| 2 | `DOC-2025-0142-10::blind::0000` | `DOC-2025-0142-10` | 0.722081 |
| 3 | `DOC-2025-0142-05::blind::0000` | `DOC-2025-0142-05` | 0.700835 |
| 4 | `DOC-2025-0142-07::blind::0000` | `DOC-2025-0142-07` | 0.700654 |
| 5 | `DOC-2025-0142-01::blind::0000` | `DOC-2025-0142-01` | 0.688269 |

**Preferred-document rank:** **1**

### Comparison

**Tie.**

Both retrieval approaches correctly rank the operative Settlement first.

---

## T10

**Query**

> What ultimately happened to the broad geographic restriction while Northstar continued protecting specific customer relationships?

**Preferred document:** `DOC-2025-0142-09` — Settlement Agreement

### Whole-Document Retrieval

| Rank | Document |
|---|---|
| 1 | `DOC-2025-0142-10` — Closing Memorandum |
| 2 | `DOC-2025-0142-09` — Settlement Agreement |

**Preferred-document rank:** **2**

### Blind Chunk Retrieval

| Rank | Chunk | Parent Document | Score |
|---|---|---|---:|
| 1 | `DOC-2025-0142-10::blind::0000` | `DOC-2025-0142-10` | 0.767257 |
| 2 | `DOC-2025-0142-02::blind::0008` | `DOC-2025-0142-02` | 0.746529 |
| 3 | `DOC-2025-0142-02::blind::0009` | `DOC-2025-0142-02` | 0.746012 |
| 4 | `DOC-2025-0142-04::blind::0000` | `DOC-2025-0142-04` | 0.735507 |
| 5 | `DOC-2025-0142-09::blind::0000` | `DOC-2025-0142-09` | 0.735105 |

**Preferred-document rank:** **5**

### Comparison

**Whole-document retrieval wins on preferred-source authority.**

However, the blind-chunk Rank-1 result — the Closing Memorandum — is still highly relevant and likely source-sufficient because it directly summarizes the ultimate resolution.

This demonstrates an important distinction:

```text
source sufficiency ≠ source authority
```

The Settlement is preferred as the operative authority, while the Closing Memorandum is a strong retrospective summary.

---

# Preferred-Document Rank Summary

| Query | Whole-Document Rank | Blind-Chunk Rank | Better Preferred-Source Rank |
|---|---:|---:|---|
| T01 | 2 | 2 | Tie |
| T02 | 2 | 2 | Tie |
| T03 | 1 | 1 | Tie |
| T04 | 2 | **1** | Blind chunks |
| T05 | 1 | 1 | Tie |
| T06 | **2** | 7 | Whole document |
| T07 | **2** | 3 | Whole document |
| T08 | 1 | 1 | Tie |
| T09 | 1 | 1 | Tie |
| T10 | **2** | 5 | Whole document |

---

# Key Findings

## 1. Whole-Document Retrieval Was More Stable for Preferred Authority

Whole-document retrieval recovered the preferred source within Top-2 for every benchmark query.

```text
Whole-document Recall@2 = 100%
```

Blind chunks did not match that even at Top-3:

```text
Blind-chunk Recall@2 = 70%
Blind-chunk Recall@3 = 80%
```

This indicates that full-document context can help preserve broader provenance and legal context.

---

## 2. Blind Chunking Improved Some Locally Specific Queries

T04 improved from:

```text
Whole document: Rank 2
Blind chunk:    Rank 1
```

The relevant Agreement clause became more semantically concentrated after removing unrelated contract text.

This demonstrates the main benefit of chunking:

```text
less irrelevant text
→ stronger local semantic signal
```

---

## 3. Blind Chunking Can Lose Broader Legal Context

T06 deteriorated from:

```text
Whole document: Rank 2
Blind chunk:    Rank 7
```

The relevant contractual meaning depends on multiple related concepts that may be split across blind boundaries.

This demonstrates the main weakness of fixed-size chunking:

```text
smaller retrieval unit
→ less noise
BUT
→ less surrounding context
```

---

## 4. Many Documents Are Semantically Overlapping

This legal matter contains repeated discussion of:

- Carter,
- Northstar,
- Vertex,
- the fifty-mile restriction,
- twelve months,
- six-month customer nonsolicitation,
- the Employment Agreement,
- the injunction,
- the Settlement.

Therefore, several documents may be simultaneously:

- semantically relevant,
- factually sufficient,
- close in embedding space.

Dense cosine similarity primarily answers:

```text
Which passage is semantically closest to the query?
```

It does **not** inherently answer:

```text
Which source is the legally preferred authority?
```

---

# Three Separate Retrieval Objectives

The experiment suggests that legal RAG evaluation should distinguish:

## 1. Semantic Relevance

Does the retrieved text discuss the subject of the query?

Embedding similarity is primarily optimized for this.

## 2. Source Sufficiency

Does the retrieved text contain enough evidence to answer the question?

Embeddings often perform well indirectly because answer-bearing text is usually semantically relevant.

## 3. Source Authority / Provenance

Is the retrieved text from the legally preferred source?

Examples:

- operative contract vs later summary,
- Settlement vs Closing Memo,
- internal legal analysis vs advocacy filing,
- original agreement vs later dispute document.

Dense cosine similarity does not inherently optimize this objective.

---

# Why Blind Chunks Can Rank a Non-Preferred Source Higher

A query may contain both:

```text
semantic subject:
customer nonsolicitation + duration
```

and:

```text
provenance qualifier:
under the original Employment Agreement
```

Several documents may strongly match the semantic subject.

The embedding model converts the entire query into one dense vector; it does not treat a phrase like:

```text
"original employment agreement"
```

as a hard symbolic filter.

Therefore:

```text
Settlement
Motion
Opposition
Closing Memo
Employment Agreement
```

can all be legitimately close in semantic space even when only one is the preferred authority.

---

# Important Limitation of the Current Chunk Benchmark

The blind-chunk benchmark currently measures:

```text
Preferred-document Recall@K
```

That means:

> Did any retrieved chunk belong to the preferred parent document?

This does **not** prove that the retrieved chunk contains the exact answer-bearing evidence.

For example:

```text
Employment Agreement chunk retrieved ✅
```

does not necessarily imply:

```text
correct Employment Agreement clause retrieved ✅
```

The next stronger evaluation should label exact gold evidence chunks and measure:

```text
Evidence Recall@1
Evidence Recall@3
Evidence Recall@5
Evidence MRR
```

---

# Next Experiment

The blind fixed-size baseline provides empirical justification for moving to **structure-aware legal chunking**.

The next chunking strategy should preserve:

- document identity,
- legal headings,
- section numbers,
- clauses,
- parent-child structure,
- page provenance,
- document type,
- source role,
- temporal / version context.

Instead of:

```text
words 1750–2050
words 2000–2300
words 2250–2550
```

the system should prefer legal units such as:

```text
§10.2 Passive Investments
§12.2 Geographic Restriction
§12.4 Competitor Employment
§12.5 Customer Non-Solicitation
```

The retrieval representation can also include contextual metadata:

```text
Document: Employment Agreement - James Carter
Document Type: Employment Agreement
Section: 12.5 Customer Non-Solicitation
Text: ...
```

This should improve semantic specificity while preserving provenance signals.

However, authority should ultimately be handled by more than chunking alone.

A production legal retrieval pipeline should eventually combine:

```text
semantic retrieval
+
metadata filtering
+
source-role awareness
+
hybrid retrieval
+
reranking
```

---

# Conclusion

The blind fixed-size chunking baseline did **not** simply outperform or underperform whole-document retrieval.

Instead, it exposed a tradeoff:

### Whole Documents

**Advantages**

- preserve broader context,
- preserve related clauses together,
- more stable preferred-source recovery.

**Disadvantages**

- unrelated text can dilute semantic signal,
- large documents are inefficient retrieval units.

### Blind Chunks

**Advantages**

- stronger local semantic concentration,
- can improve exact-clause retrieval,
- smaller context units.

**Disadvantages**

- can separate related legal concepts,
- can lose headings / structure,
- can reduce provenance clarity,
- semantically similar non-authoritative documents may outrank the preferred source.

The strongest current finding is:

> **Whole-document retrieval was better for preferred authoritative-source stability, while blind chunk retrieval could be better for locally concentrated evidence.**

This motivates the next experiment:

> **structure-aware legal chunking followed by evidence-level retrieval evaluation.**
