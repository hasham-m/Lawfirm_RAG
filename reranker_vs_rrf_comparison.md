# RRF vs Cohere Reranker — Retrieval Experiment Summary

## What changed

**RRF** combines the rankings from Dense V3 and BM25. It rewards candidates that are ranked well by one or both retrievers, but it does not deeply evaluate whether a chunk directly answers the query.

**Cohere reranking** takes the RRF Top 20 and re-evaluates each candidate together with the exact query. It then assigns a new relevance score and produces a new ordering.

```text
Dense V3 + BM25
      ↓
     RRF
      ↓
  Top 20 candidates
      ↓
 Cohere Reranker
      ↓
 more precise relevance ordering
```

## Query-by-query comparison

| Query | Preferred evidence/source | RRF Rank | Reranker Rank | Result |
|---|---|---:|---:|---|
| T01 | Employment Agreement | 2 | 3* | Slight regression |
| T02 | Preliminary Legal Analysis | 1 | 1 | Maintained |
| T03 | Hearing Notes | 1 | 1 | Maintained |
| T04 | Employment Agreement §12.4 | 1 | 1 | Maintained |
| T05 | Employment Agreement §10.2 | 1 | 1 | Maintained |
| T06 | Employment Agreement §12.5 | 2 | 2 | Maintained strong result |
| T07 | Preliminary Legal Analysis | 7 | **2** | **Major improvement** |
| T08 | Motion for Preliminary Injunction | 1 | 1 | Maintained |
| T09 | Settlement Agreement §2 | 3 | **2** | **Improvement** |
| T10 | Settlement Agreement §3 | 4 | 8 | Regression |

\*For T01, the preferred Employment Agreement begins at reranker rank 3, but the correct 12-month / 50-mile answer is still available in the Top 2 through the Client Intake Memorandum.

## Main improvements

The clearest reranker win was **T07**, where the preferred Preliminary Legal Analysis moved from RRF rank **7 → 2**. The reranker better understood that the question was asking which restriction was considered *more defensible*, instead of only rewarding literal geographic-term overlap.

**T09** also improved from **3 → 2**, moving the operative Settlement Agreement closer to the top.

For **T04, T05, and T08**, RRF was already excellent and the reranker correctly preserved the preferred evidence at rank 1.

For **T06**, the exact answer-bearing clause §12.5 remained at rank 2. The big gain for T06 had already happened during RRF, which moved the clause from rank 4 in Dense V3/BM25 to rank 2.

## Main failure: T10

T10 is the important remaining failure.

The reranker moved the earlier Preliminary Legal Analysis to rank 1 while the preferred Settlement §3 dropped from RRF rank 4 to reranker rank 8.

This shows the limit of pure relevance reranking:

```text
Reranker:
Which passage most directly matches the query?

Authority logic:
Which source should control the answer?
```

The Preliminary Legal Analysis is highly relevant to the wording of T10, but it is not the final operative resolution. The Settlement Agreement is the better legal source for what **ultimately happened**.

## Conclusion

The reranker substantially improved **precision** after RRF.

- Strict preferred-source/evidence Top-2 performance: **8/10 queries**
- If answer-bearing evidence is counted for T01: **9/10 queries**
- Strongest improvement: **T07, rank 7 → 2**
- Additional improvement: **T09, rank 3 → 2**
- Main remaining failure: **T10, rank 4 → 8**

The experiment confirms that reranking is useful for selecting the most directly relevant chunks, but it does **not replace legal source-role / authority logic**.
