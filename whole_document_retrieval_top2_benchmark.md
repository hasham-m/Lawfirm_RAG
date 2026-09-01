# Whole-Document Retrieval Benchmark — Top-2 Results

## Objective

This document records the **retrieval-only** results for the 10 frozen benchmark queries in the Hamilton & Cole LLP / Carter matter corpus.

The purpose is to evaluate the embedding + cosine-similarity retrieval layer **before any LLM generation is considered**.

For each query, the table records:

- **Gold Answer** — the expected factual answer.
- **Preferred Source** — the document considered the strongest or most appropriate evidentiary source for the query.
- **Rank 1** — the highest-cosine-similarity retrieved document.
- **Rank 2** — the second-highest-cosine-similarity retrieved document.
- **Gold Rank** — the retrieval rank at which the preferred source appears.

> **Important:** “Preferred Source” is an evaluation target for source quality/authority. A different retrieved document may still be factually sufficient for some questions.

---

## Retrieval Results

| ID | Query | Gold Answer | Preferred Source | Rank 1 | Rank 2 | Gold Rank |
|---|---|---|---|---|---|---:|
| **T01** | How long and how far did Carter's original non-compete restrict him? | **12 months** and **within 50 miles of the Company's Westbridge office**. | **Employment Agreement - James Carter** (`DOC-2025-0142-02`) | Carter Opposition to Preliminary Injunction (`DOC-2025-0142-07`) — **0.715065** | Employment Agreement - James Carter (`DOC-2025-0142-02`) — **0.704227** | **2** |
| **T02** | Why was Carter's fifty-mile non-compete potentially difficult to enforce? | Carter worked with **national accounts**, so a 50-mile geographic restriction did not closely track his actual customer territory or competitive risk and was vulnerable to being narrowed or denied. | **Restrictive Covenant - Preliminary Legal Analysis** (`DOC-2025-0142-04`) | Carter Opposition to Preliminary Injunction (`DOC-2025-0142-07`) — **0.719031** | Restrictive Covenant - Preliminary Legal Analysis (`DOC-2025-0142-04`) — **0.710534** | **2** |
| **T03** | What did Judge Maren think about the fifty-mile geographic restriction? | Judge Maren repeatedly questioned why a **50-mile geographic limit** made sense for an employee servicing national accounts, while noting that the **12-month duration did not appear facially excessive**. | **Preliminary Injunction Hearing Notes** (`DOC-2025-0142-08`) | Preliminary Injunction Hearing Notes (`DOC-2025-0142-08`) — **0.675471** | Carter Opposition to Preliminary Injunction (`DOC-2025-0142-07`) — **0.648143** | **1** |
| **T04** | Under the original employment agreement, did simply accepting employment with a competitor automatically violate the agreement? | **No.** Competitor employment was not automatically prohibited; Carter could remain employed by a competitor if his duties fell outside the restricted competitive services / territory or were structured to avoid use or disclosure of confidential information. | **Employment Agreement - James Carter** (`DOC-2025-0142-02`) | Client Intake Memorandum (`DOC-2025-0142-01`) — **0.693041** | Employment Agreement - James Carter (`DOC-2025-0142-02`) — **0.684456** | **2** |
| **T05** | What percentage of a publicly traded competitor could Carter own without violating the agreement solely because of the investment? | **Less than 2%**, provided the ownership was a passive investment. | **Employment Agreement - James Carter** (`DOC-2025-0142-02`) | Employment Agreement - James Carter (`DOC-2025-0142-02`) — **0.703075** | Settlement Agreement and Mutual Release (`DOC-2025-0142-09`) — **0.701683** | **1** |
| **T06** | How long was Carter prohibited from directly soliciting customers he had been responsible for before leaving Northstar? | **Six months after separation** under the original Employment Agreement for customers for whom he had material responsibility. | **Employment Agreement - James Carter** (`DOC-2025-0142-02`) | Settlement Agreement and Mutual Release (`DOC-2025-0142-09`) — **0.785051** | Employment Agreement - James Carter (`DOC-2025-0142-02`) — **0.766773** | **2** |
| **T07** | What restriction was considered more defensible than preventing Carter from competing within fifty miles of Westbridge? | The **six-month customer non-solicitation covenant** was considered materially narrower and more defensible than the broad geographic non-compete. | **Restrictive Covenant - Preliminary Legal Analysis** (`DOC-2025-0142-04`) | Carter Opposition to Preliminary Injunction (`DOC-2025-0142-07`) — **0.728537** | Restrictive Covenant - Preliminary Legal Analysis (`DOC-2025-0142-04`) — **0.703161** | **2** |
| **T08** | What did Northstar actually ask the court to stop Carter from doing at Vertex? | Northstar asked the court to prohibit Carter from providing **substantially similar enterprise-sales services to Vertex within 50 miles of Westbridge for the remainder of the 12-month restricted period**, and to enforce the **six-month customer non-solicitation covenant**. | **Motion for Preliminary Injunction** (`DOC-2025-0142-06`) | Motion for Preliminary Injunction (`DOC-2025-0142-06`) — **0.779587** | Carter Opposition to Preliminary Injunction (`DOC-2025-0142-07`) — **0.769103** | **1** |
| **T09** | After the dispute was resolved, was Carter still allowed to work for Vertex? | **Yes.** Carter was permitted to remain employed by Vertex, and Northstar agreed not to seek enforcement of the competitive restriction solely because of that employment. | **Settlement Agreement and Mutual Release** (`DOC-2025-0142-09`) | Settlement Agreement and Mutual Release (`DOC-2025-0142-09`) — **0.744242** | Matter Closing Memorandum (`DOC-2025-0142-10`) — **0.728162** | **1** |
| **T10** | What ultimately happened to the broad geographic restriction while Northstar continued protecting specific customer relationships? | The broad geographic restriction was **not ultimately used to bar Carter from remaining at Vertex**; Northstar instead preserved targeted customer non-solicitation protections and confidentiality obligations. | **Settlement Agreement and Mutual Release** (`DOC-2025-0142-09`) | Matter Closing Memorandum (`DOC-2025-0142-10`) — **0.751536** | Settlement Agreement and Mutual Release (`DOC-2025-0142-09`) — **0.735456** | **2** |

---

## Retrieval Summary

Using the rankings recorded in this Top-2 experiment:

- **Preferred source at Rank 1:** 4 / 10 queries
- **Recall@1:** **40%**
- **Preferred source within Top 2:** 10 / 10 queries
- **Recall@2:** **100%**
- **Mean Reciprocal Rank (MRR):** **0.70**

### Rank Distribution

| Gold Rank | Query Count |
|---|---:|
| Rank 1 | 4 |
| Rank 2 | 6 |
| Rank 3+ | 0 |

---

## Key Retrieval Findings

1. **Top-1 retrieval was insufficient as the sole retrieval depth.**  
   The preferred source appeared first for only 4 of 10 queries.

2. **Top-2 retrieval materially improved source recall.**  
   In this recorded run, every preferred source appeared within the first two retrieved documents.

3. **Semantic similarity did not consistently track legal source authority.**  
   Advocacy filings, intake memoranda, and closing summaries sometimes ranked above contracts, legal analyses, or operative settlement documents.

4. **A high cosine-similarity score should not be treated as legal confidence.**  
   The highest-scoring document may be semantically close to the query while still being a weaker evidentiary source.

5. **Retrieval quality and generator source-selection quality must be evaluated separately.**  
   Retrieval determines whether the strong source is available in context; the generator must still recognize and select it appropriately.

---

## Benchmark Interpretation Note

This file evaluates **retrieval only**. It does not score Gemini, GPT-OSS, Qwen, or any other generator.

A separate generation experiment should evaluate:

- answer correctness,
- groundedness,
- preferred-source selection,
- source-role classification,
- page/section localization,
- abstention behavior,
- hallucination.

This separation prevents retrieval performance from being confused with downstream LLM reasoning quality.
