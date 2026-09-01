# Whole-Document RAG Experiment — Gemini Top-1 vs Top-2 Generation

## Overview

This report compares the downstream behavior of the **same Gemini generator** when given:

- **Top-1 whole-document retrieval context**, versus
- **Top-2 whole-document retrieval context**

for the frozen Carter / Northstar Analytics benchmark.

The purpose is not merely to ask whether the final answer was correct. The comparison evaluates whether increasing retrieval depth changed:

- answerability,
- completeness,
- grounding,
- preferred-source selection,
- source-role awareness,
- evidence localization,
- and abstention behavior.

The underlying retrieval approach remained the same:

```text
query
→ Gemini query embedding
→ cosine similarity against saved document embeddings
→ ranked whole documents
→ Top-K context
→ Gemini structured generation
```

The generation schema remained focused on:

- `answer`
- `source_document`
- `document_id`
- `page`
- `section`
- `supporting_evidence`
- `source_sufficiency`
- `source_role`
- `authority_note`

---

# 1. Experimental Control

For the clean comparison, the generator is held constant:

```text
Gemini + Top-1
vs
Gemini + Top-2
```

This allows retrieval depth to be the principal changed variable.

## Clean comparison set

The supplied records support a clean Gemini-vs-Gemini comparison for:

- T01
- T02
- T03
- T04
- T05
- T06
- T07
- T08
- T09

T10 is discussed separately because the supplied Top-2 experiment file also contains GPT-OSS / Groq runs for that query, making generator identity in the final T10 records unsuitable for inclusion in the strict Gemini-only aggregate.

---

# 2. Executive Summary

Across the clean T01–T09 comparison:

| Metric | Gemini Top-1 | Gemini Top-2 |
|---|---:|---:|
| Direct substantive answers | **6 / 9** | **9 / 9** |
| Correct partial answers | **1 / 9** | **0 / 9** |
| Correct abstentions caused by insufficient context | **2 / 9** | **0 / 9** |
| Preferred-source selections | **4 / 9** | **9 / 9** |
| Obvious hallucinated unseen-source provenance | **0** | **0** |

### Main result

> Increasing retrieval depth from Top-1 to Top-2 materially improved downstream answerability **without requiring the generator to become more speculative**.

The most important repairs occurred on:

- **T01** — partial answer became complete.
- **T04** — correct abstention became a correct contractual answer.
- **T07** — correct abstention became a correct legal-analysis answer.

Top-2 also improved **source quality / provenance** on:

- **T02** — advocacy filing → legal research memo.
- **T06** — settlement-based answer → original Employment Agreement.

---

# 3. Query-by-Query Comparison

## T01 — Original non-compete duration and geography

**Query**

> How long and how far did Carter's original non-compete restrict him?

**Gold answer**

- Duration: **12 months**
- Geography: **within 50 miles of the Company's Westbridge office**

### Top-1

**Retrieved document**

`DOC-2025-0142-07` — Carter Opposition to Preliminary Injunction

**Gemini behavior**

The Opposition supplied the 50-mile restriction but did not supply the duration.

Gemini therefore:

- answered the geography,
- explicitly acknowledged the missing duration,
- classified the context as `partially_sufficient`,
- and did **not hallucinate** the unseen 12-month term.

**Evaluation**

✅ Grounded partial answer  
✅ Correct abstention on missing component  
❌ Full answer unavailable  
❌ Preferred source unavailable at Top-1

### Top-2

The Employment Agreement appeared as Rank 2:

`DOC-2025-0142-02` — Employment Agreement - James Carter

Gemini selected it rather than Rank 1 and returned:

- **12 months**
- **50 miles**
- Page **5**
- Section **12. Post-Employment Restrictive Covenants**
- `source_sufficiency = sufficient`
- `source_role = operative_source`

**Evaluation**

✅ Complete answer  
✅ Correct preferred source  
✅ Correct authority reasoning  
✅ Correct evidence localization

### Change

```text
Top-1:
partial answer because duration was absent

Top-2:
complete answer from original Employment Agreement
```

### Lesson

> A grounded LLM cannot recover a fact that retrieval never supplied. Increasing K repaired the failure by exposing the missing operative document, after which the generator correctly preferred Rank 2 over the semantically higher-ranked advocacy filing.

---

## T02 — Why the 50-mile restriction was difficult to enforce

**Query**

> Why was Carter's fifty-mile non-compete potentially difficult to enforce?

**Gold answer**

Carter serviced national accounts, so the 50-mile restriction did not closely track the actual scope of his work or competitive risk.

### Top-1

**Retrieved document**

`DOC-2025-0142-07` — Carter Opposition to Preliminary Injunction

Gemini correctly explained the national-account mismatch.

It classified:

- `source_sufficiency = sufficient`
- `source_role = advocacy`

The answer was factually grounded even though the source was not the preferred analytical source.

### Top-2

Rank 2 added:

`DOC-2025-0142-04` — Restrictive Covenant - Preliminary Legal Analysis

Gemini selected the legal research memo and returned the same central answer, now grounded in the document that directly performs the enforceability analysis.

It classified:

- Page **1**
- Section **Analysis**
- `source_sufficiency = sufficient`
- `source_role = internal_analysis`

### Change

```text
Top-1:
correct answer from advocacy

Top-2:
correct answer from preferred legal analysis
```

### Lesson

> More retrieval depth does not always change the factual answer. Sometimes it improves **source authority, neutrality, and provenance** instead.

This is a key distinction:

```text
answer correctness
≠
source quality
```

---

## T03 — Judge Maren's view of the geographic restriction

**Query**

> What did Judge Maren think about the fifty-mile geographic restriction?

**Gold answer**

Judge Maren questioned why a 50-mile geographic limit made sense for an employee servicing national accounts, while noting that the 12-month duration did not appear facially excessive.

### Top-1

**Retrieved document**

`DOC-2025-0142-08` — Preliminary Injunction Hearing Notes

Gemini answered correctly and recognized the notes as `internal_analysis`.

### Top-2

Rank 1 remained the Hearing Notes, with Carter's Opposition added as Rank 2.

Gemini continued to select:

`DOC-2025-0142-08` — Preliminary Injunction Hearing Notes

and localized:

- Page **1**
- Section **Court Observations**

### Change

No material answer change.

### Lesson

> Increasing K should not force the generator to switch sources. When Rank 1 is already the strongest source, a capable generator should remain stable despite additional context.

This is a useful **non-regression case**.

---

## T04 — Whether competitor employment automatically violated the agreement

**Query**

> Under the original employment agreement, did simply accepting employment with a competitor automatically violate the agreement?

**Gold answer**

**No.**

### Top-1

**Retrieved document**

`DOC-2025-0142-01` — Client Intake Memorandum

The Intake Memo did not contain the precise contractual exception.

Gemini therefore returned:

- `source_sufficiency = insufficient`
- `source_role = internal_analysis`

and correctly refused to infer an unseen contractual term.

**Evaluation**

✅ Excellent abstention  
✅ No hallucinated contract clause  
❌ Unable to answer from available context

### Top-2

Rank 2 added:

`DOC-2025-0142-02` — Employment Agreement - James Carter

Gemini selected Rank 2 and answered:

> No. Accepting employment with a competitor did not automatically violate the agreement.

Localization:

- Page **5**
- Section **12.4**
- `source_sufficiency = sufficient`
- `source_role = operative_source`

### Change

```text
Top-1:
INSUFFICIENT → abstain

Top-2:
SUFFICIENT → correct contractual answer
```

### Lesson

> This is one of the clearest demonstrations that retrieval depth can directly control downstream answerability.

The important success is not that Top-1 "failed." The important success is that Top-1 **failed safely**, while Top-2 supplied the missing authority and converted the safe abstention into a grounded answer.

---

## T05 — Passive-investment threshold

**Query**

> What percentage of a publicly traded competitor could Carter own without violating the agreement solely because of the investment?

**Gold answer**

**Less than 2%.**

### Top-1

Rank 1 was already:

`DOC-2025-0142-02` — Employment Agreement - James Carter

Gemini correctly returned:

- less than 2%,
- Page **4**,
- Section **10.2**,
- `source_sufficiency = sufficient`,
- `source_role = operative_source`.

### Top-2

The Settlement Agreement was added as Rank 2.

Gemini still selected the Employment Agreement and returned the same narrow contractual fact.

### Change

No material change.

### Lesson

> Top-2 did not degrade a query that was already correctly served by Top-1.

This demonstrates that more context can be harmless when the generator correctly distinguishes relevant from irrelevant additional evidence.

---

## T06 — Customer non-solicitation duration

**Query**

> How long was Carter prohibited from directly soliciting customers he had been responsible for before leaving Northstar?

**Gold benchmark answer**

**Six months after separation under the original Employment Agreement.**

### Top-1

Rank 1 was:

`DOC-2025-0142-09` — Settlement Agreement and Mutual Release

Gemini answered from the Settlement Agreement and correctly identified a six-month customer restriction.

This was grounded, but the benchmark query itself was ambiguous because it did not explicitly say:

> "Under the original Employment Agreement..."

Therefore the Top-1 answer was defensible for the literal question even though it did not use the intended original-contract source.

### Top-2

In the current Top-2 run, Rank 2 was:

`DOC-2025-0142-02` — Employment Agreement - James Carter

Gemini selected the Employment Agreement and returned:

- **six months after separation**
- Page **5**
- Section **12.5**
- `source_sufficiency = sufficient`
- `source_role = operative_source`

It also noted that the later Settlement reinforced the six-month restriction.

### Change

```text
Top-1:
correct literal answer from later Settlement

Top-2:
correct answer from intended original Employment Agreement
```

### Lesson

> Top-2 improved **provenance and benchmark alignment**, not merely factual accuracy.

This query also revealed a benchmark-design flaw: questions that can be independently answered by multiple operative documents should explicitly anchor the intended time period or agreement when provenance matters.

---

## T07 — More defensible restriction

**Query**

> What restriction was considered more defensible than preventing Carter from competing within fifty miles of Westbridge?

**Gold answer**

The **six-month customer non-solicitation covenant**.

### Top-1

**Retrieved document**

`DOC-2025-0142-07` — Carter Opposition to Preliminary Injunction

The Opposition showed Carter requesting narrower relief, but it did not establish the analytical conclusion that a particular restriction was considered more defensible.

Gemini therefore returned:

- `source_sufficiency = insufficient`
- `source_role = advocacy`

and correctly abstained.

### Top-2

Rank 2 added:

`DOC-2025-0142-04` — Restrictive Covenant - Preliminary Legal Analysis

Gemini selected the memo and answered:

> The six-month customer non-solicitation covenant was materially narrower and presented the stronger enforcement argument.

Localization:

- Page **1**
- Section **Short Answer**
- `source_sufficiency = sufficient`
- `source_role = internal_analysis`

### Change

```text
Top-1:
INSUFFICIENT → abstain

Top-2:
SUFFICIENT → correct analytical conclusion
```

### Lesson

> Semantic similarity can retrieve a document containing related advocacy without retrieving the document that actually establishes the analytical proposition being asked.

Top-2 repaired that provenance failure.

---

## T08 — Relief Northstar requested from the court

**Query**

> What did Northstar actually ask the court to stop Carter from doing at Vertex?

**Gold answer**

Northstar asked the court to prohibit substantially similar enterprise-sales work at Vertex within 50 miles of Westbridge for the remainder of the 12-month period and to enforce the six-month customer non-solicitation covenant.

### Top-1

Rank 1:

`DOC-2025-0142-06` — Motion for Preliminary Injunction

Gemini answered correctly.

Although the Motion is advocacy, it is the direct source for the proposition:

> What did Northstar ask the court to do?

### Top-2

Carter's Opposition was added as Rank 2.

Gemini remained with the Motion and localized:

- Page **1**
- Section **Relief Requested**
- `source_sufficiency = sufficient`
- `source_role = advocacy`

### Change

No material answer change.

### Lesson

> Source type must be evaluated relative to the question. An advocacy filing can be the best source when the question asks what that party requested.

Top-2 correctly preserved the direct source instead of blindly preferring a supposedly more neutral document.

---

## T09 — Whether Carter could remain at Vertex

**Query**

> After the dispute was resolved, was Carter still allowed to work for Vertex?

**Gold answer**

**Yes.**

### Top-1

Rank 1:

`DOC-2025-0142-09` — Settlement Agreement and Mutual Release

Gemini correctly answered yes and localized:

- Page **1**
- Section **2. Employment at Vertex**
- `source_sufficiency = sufficient`
- `source_role = operative_source`

### Top-2

Rank 2 added:

`DOC-2025-0142-10` — Matter Closing Memorandum

Gemini continued to select the Settlement Agreement and explicitly recognized the Closing Memo as secondary confirmation.

### Change

No material answer change.

### Lesson

> When both an operative agreement and a later internal summary are available, the generator can preserve the operative source rather than selecting the easier summary.

This is a strong source-authority result.

---

# 4. Clean Gemini Comparison Scorecard

| Query | Top-1 behavior | Top-2 behavior | Preferred source selected at Top-1? | Preferred source selected at Top-2? | Main effect of Top-2 |
|---|---|---|---:|---:|---|
| **T01** | Partial | Complete | No | Yes | Recovered missing duration + authoritative contract |
| **T02** | Correct from advocacy | Correct from legal analysis | No | Yes | Improved source quality / authority |
| **T03** | Correct | Correct | Yes | Yes | Stable / no regression |
| **T04** | Correct abstention | Complete | No | Yes | Recovered missing contractual rule |
| **T05** | Correct | Correct | Yes | Yes | Stable / no regression |
| **T06** | Correct literal answer from Settlement | Correct from original Agreement | No | Yes | Improved provenance / benchmark alignment |
| **T07** | Correct abstention | Complete | No | Yes | Recovered missing analytical conclusion |
| **T08** | Correct | Correct | Yes | Yes | Stable / no regression |
| **T09** | Correct | Correct | Yes | Yes | Stable / no regression |

---

# 5. Aggregate Interpretation

## 5.1 Answerability

### Top-1

Across T01–T09:

- Direct substantive answers: **6 / 9**
- Correct partial answers: **1 / 9**
- Correct abstentions: **2 / 9**

The two abstentions were not generator failures. They were appropriate responses to missing evidence.

### Top-2

Across T01–T09:

- Direct substantive answers: **9 / 9**
- Partial answers: **0 / 9**
- Abstentions caused by insufficient retrieved context: **0 / 9**

### Result

> Top-2 transformed all three context-limited Top-1 cases into complete grounded answers.

---

## 5.2 Preferred-source selection

### Top-1

Preferred source selected:

- T03
- T05
- T08
- T09

**4 / 9**

### Top-2

Preferred source selected:

- T01
- T02
- T03
- T04
- T05
- T06
- T07
- T08
- T09

**9 / 9**

### Result

```text
Preferred-source selection:
44.4% → 100%
```

within the clean nine-query Gemini comparison.

This is a major result because Top-2 did not merely give the LLM "more text." It often gave it the **better legal source**.

---

# 6. Core Lessons

## Lesson 1 — Retrieval recall and generation quality are separate

A generator cannot cite or reason from a document that retrieval did not provide.

T01, T04, and T07 demonstrate:

```text
missing source at Top-1
→ partial answer or abstention

source becomes available at Top-2
→ complete grounded answer
```

---

## Lesson 2 — Safe abstention is a success mode

Top-1 did not simply produce "worse answers."

For T04 and T07, Gemini correctly recognized that the retrieved document could not establish the proposition.

That behavior is preferable to hallucinating the expected answer.

Therefore:

```text
Top-1 abstention
≠ generator failure
```

It reveals a retrieval limitation while preserving grounding.

---

## Lesson 3 — More context can improve source authority without changing the answer

T02 was already answerable from the Opposition.

Top-2 improved the source from:

```text
advocacy filing
→ internal legal analysis
```

The factual answer remained substantially the same.

Thus RAG evaluation should separately score:

- answer correctness,
- source sufficiency,
- source authority / appropriateness.

---

## Lesson 4 — Higher cosine similarity does not equal stronger legal authority

Several Rank-1 documents were semantically closer to the question but weaker as evidentiary sources.

Examples:

- T01: Opposition outranked Employment Agreement.
- T02: Opposition outranked Legal Analysis.
- T04: Intake Memo outranked Employment Agreement.
- T07: Opposition outranked Legal Analysis.

The generator must therefore perform a second-stage judgment:

```text
semantic relevance
≠
legal authority
```

---

## Lesson 5 — Top-2 can improve provenance even when Top-1 is factually correct

T06 is the clearest example.

Top-1 Settlement supplied a defensible six-month answer.

Top-2 exposed the original Employment Agreement, allowing Gemini to anchor the answer to the intended original contractual provision.

Thus:

```text
correct fact
≠
correct provenance
```

---

## Lesson 6 — Additional context did not cause obvious regressions

Queries already well-served by Rank 1 remained stable:

- T03
- T05
- T08
- T09

This matters because increasing K has a potential downside:

```text
more context
→ more distractors
→ possible source confusion
```

In these cases, Gemini successfully ignored the weaker additional document.

---

# 7. Important Benchmark Caveat — T06

T06 is not a perfectly isolated provenance test.

The question:

> How long was Carter prohibited from directly soliciting customers he had been responsible for before leaving Northstar?

can be answered from both:

- the original Employment Agreement, and
- the later Settlement Agreement.

Therefore a Settlement-based answer is not necessarily hallucinated or factually wrong.

For the current frozen benchmark, T06 should remain unchanged for comparability.

For a future benchmark revision, use wording such as:

> Under Carter's original Employment Agreement, how long was he prohibited from directly soliciting Restricted Customers after separation?

---

# 8. Supplemental T10 — Mixed-Generator Record

**Query**

> What ultimately happened to the broad geographic restriction while Northstar continued protecting specific customer relationships?

### Top-1 Gemini

Rank 1 was:

`DOC-2025-0142-10` — Matter Closing Memorandum

Gemini answered correctly from the retrospective Closing Memo and classified it as internal analysis.

The result demonstrated that:

> a secondary/internal summary can be sufficient to answer a retrospective factual question even when it is not the operative instrument.

### Top-2 supplied records

The supplied Top-2 experiment records also include GPT-OSS / Groq runs for T10.

Those runs are useful for **generator-sensitivity analysis**, but they should not be counted inside the strict Gemini Top-1 vs Gemini Top-2 aggregate unless the Gemini identity of the selected T10 Top-2 record is independently confirmed.

### Why this matters

If the generator changes at the same time as K:

```text
Top-1 Gemini
vs
Top-2 GPT-OSS
```

then two variables changed:

1. retrieval depth,
2. generator model.

A difference in output can no longer be attributed cleanly to Top-K alone.

### Lesson

> Controlled RAG experiments should change one major variable at a time.

T10 is still valuable, but it belongs either in:

- a confirmed Gemini Top-2 rerun, or
- a separate Gemini-vs-GPT-OSS generator comparison.

---

# 9. What the Top-1 vs Top-2 Experiment Demonstrates

The strongest conclusion from this experiment is:

> **Increasing retrieval depth from one whole document to two materially improved downstream answerability and source quality, while the generator remained grounded and continued to abstain when evidence was absent.**

More specifically:

```text
Top-1:
retrieval frequently exposed semantically relevant
but non-preferred sources.

Top-2:
the preferred legal source became available for
the clean comparison set.

Gemini:
often recognized that Rank 2 was stronger than Rank 1
and selected it.
```

Therefore the pipeline exposes two distinct responsibilities:

```text
RETRIEVER
→ make the right evidence available

GENERATOR
→ choose and reason from the right evidence
```

Neither layer can be evaluated solely through final-answer accuracy.

---

# 10. Implications for the Next Experiment

Whole-document Top-2 solved many answerability problems, but it does not solve several production concerns:

- large prompt size,
- poor evidence granularity,
- high LLM context cost,
- increased latency,
- multiple unrelated sections inside one retrieved document,
- and the need for deterministic citation metadata.

The next experiment should therefore move to:

```text
blind / fixed-size chunk retrieval
```

and compare:

- retrieval recall,
- answer correctness,
- evidence localization,
- context tokens,
- latency,
- and source reconstruction

against the whole-document baseline.

The purpose is not to assume chunking is better.

The purpose is to measure:

> whether reducing context granularity preserves or improves retrieval accuracy while lowering generation cost and improving evidence localization.
