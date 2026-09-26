# Structure-Aware V3 — Small Ranking Improvements

V3 did **not materially improve the preferred-source benchmark rankings** over V2, but a few individual contract chunks moved upward after the definition context was made more focused.

| Query | Chunk | V2 Rank | V3 Rank | Change |
|---|---|---:|---:|---:|
| T03 — Judge Maren / fifty-mile restriction | Employment Agreement §12.2 | 5 | 2 | +3 |
| T04 — competitor employment automatically violating agreement | Employment Agreement §12.1 | 4 | 2 | +2 |
| T05 — passive investment percentage | Employment Agreement §12.4 | 5 | 4 | +1 |
| T07 — more defensible restriction than fifty-mile non-compete | Employment Agreement §12.2 | 2 | 1 | +1 |

## Takeaway

V3 made some Employment Agreement vectors more focused by embedding only bounded, actual defined-term chains.

However, the **preferred-source rankings mostly stayed the same**, which suggests that the remaining failures are not mainly caused by definition enrichment anymore.

The next step is:

`BM25 / sparse retrieval -> hybrid retrieval -> reranking`
