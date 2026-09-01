# Northstar Whole-Document RAG Experiment
## Experiment 2B.1-A — Top-1 Whole Document → LLM

**Matter:** HC-2025-0142 — Carter / Northstar Analytics  
**Experiment date:** August 29, 2026  
**Corpus:** 10 whole documents  
**Retrieval:** Gemini query embedding + manual NumPy cosine similarity  
**Generation:** One retrieved document only  
**Output validation:** Pydantic structured response  

---

## 1. Purpose

This experiment tests whether a generative LLM can answer and localize evidence when it receives only the **Top-1 whole document** returned by semantic retrieval.

The experiment intentionally separates three questions:

1. **Retrieval:** Did whole-document cosine similarity return the preferred authoritative or operative source?
2. **Generation:** Given only the retrieved Top-1 document, could the LLM answer the question from that document?
3. **Source awareness:** Did the LLM correctly distinguish a document that contains useful facts from a document that directly establishes the proposition being asked about?

The LLM was not given:
- the gold answer,
- the expected source,
- the other nine documents,
- retrieval rankings beyond Top-1,
- or any document embeddings.

It received only:
- the original English query,
- metadata for the Top-1 document,
- page-labelled English text for the Top-1 document,
- and grounding/source-classification instructions.

---

## 2. Structured Output Schema

Every response was constrained through Pydantic to contain:

- `answer`
- `source_document`
- `document_id`
- `page`
- `section`
- `supporting_evidence`
- `source_sufficiency`
- `source_role`
- `authority_note`

Allowed `source_sufficiency` values:

- `sufficient`
- `partially_sufficient`
- `insufficient`

Allowed `source_role` values:

- `primary_source`
- `operative_source`
- `internal_analysis`
- `advocacy`
- `secondary_summary`
- `other`
- `unclear`

---

## 3. Prior Whole-Document Retrieval Baseline

The preferred authoritative / operative source ranked:

| Query | Preferred source rank |
|---|---:|
| T01 | 2 |
| T02 | 2 |
| T03 | 1 |
| T04 | 2 |
| T05 | 1 |
| T06 | 3 |
| T07 | 2 |
| T08 | 1 |
| T09 | 1 |
| T10 | 2 |

Therefore, strict preferred-source Top-1 retrieval accuracy was **4/10 (40%)**.

This is important because the LLM had the preferred source available for only four of the ten questions.

---

# 4. Top-1 → LLM Results

## T01 — Original non-compete duration and geography

**Query:**  
How long and how far did Carter's original non-compete restrict him?

**Top-1 retrieval:**  
DOC-2025-0142-07 — Carter Opposition to Preliminary Injunction

**Cosine similarity:** `0.715065`

**Gold answer:**  
12 months; within 50 miles of Northstar's Westbridge office.

**LLM result:**  
The LLM found the 50-mile restriction but correctly stated that the Opposition did not provide the duration.

**LLM classification:**
- Source sufficiency: `partially_sufficient`
- Source role: `advocacy`

**Evaluation:** **Strong grounded partial success**

The model did not invent the missing 12-month duration. This is exactly the desired behavior when Top-1 contains only part of the requested answer.

**Key lesson:**  
The retriever missed the original Employment Agreement, and the LLM did **not** hide that retrieval failure by hallucinating the missing contractual term.

---

## T02 — Why the 50-mile restriction was difficult to enforce

**Query:**  
Why was Carter's fifty-mile non-compete potentially difficult to enforce?

**Top-1 retrieval:**  
DOC-2025-0142-07 — Carter Opposition to Preliminary Injunction

**Cosine similarity:** `0.719031`

**Gold answer:**  
Carter handled national accounts, making the 50-mile geographic restriction poorly matched to the actual scope of his work / competitive risk.

**LLM result:**  
The LLM answered that the restriction was disconnected from Carter's national customer territory, broader than necessary, and that Vertex assigned him to a different product segment.

**LLM classification:**
- Source sufficiency: `sufficient`
- Source role: `advocacy`

**Evaluation:** **Correct and grounded**

Originally, this experiment expected `partially_sufficient` because the Opposition was not the preferred internal legal-analysis source. That expectation was too strict.

The Opposition itself contains enough evidence to answer **why the restriction was argued to be difficult to enforce**. Its advocacy status affects source role and neutrality, but does not automatically make the factual support incomplete.

**Key lesson:**  

> Source sufficiency is not the same thing as source authority or neutrality.

An advocacy document can be fully sufficient for a question about the argument it contains.

---

## T03 — Judge Maren's view of the 50-mile restriction

**Query:**  
What did Judge Maren think about the fifty-mile geographic restriction?

**Top-1 retrieval:**  
DOC-2025-0142-08 — Preliminary Injunction Hearing Notes

**Cosine similarity:** `0.675471`

**Gold answer:**  
Judge Maren repeatedly questioned why a 50-mile geographic restriction made sense for an employee servicing national accounts. The 12-month duration itself did not appear facially excessive.

**LLM result:**  
Correct.

**LLM classification:**
- Source sufficiency: `sufficient`
- Source role: `internal_analysis`

**Evaluation:** **Clean success**

The model also made an important source distinction: the hearing notes are internal attorney work-product notes reporting what happened at the hearing, not an official judicial transcript or order.

This is more precise than labeling the notes a generic `primary_source`.

**Key lesson:**  
The LLM was capable of answering correctly while still recognizing the evidentiary character of the retrieved document.

---

## T04 — Whether competitor employment automatically violated the original agreement

**Query:**  
Under the original employment agreement, did simply accepting employment with a competitor automatically violate the agreement?

**Top-1 retrieval:**  
DOC-2025-0142-01 — Client Intake Memorandum

**Cosine similarity:** `0.693041`

**Gold answer:**  
No. The original Employment Agreement did not automatically prohibit competitor employment.

**LLM result:**  
The LLM refused to infer the precise contractual rule from the Client Intake Memorandum.

**LLM classification:**
- Source sufficiency: `insufficient`
- Source role: `internal_analysis`

**Evaluation:** **Excellent abstention**

The Client Intake Memorandum says only that the agreement *appears* to contain a non-compete and other restrictions. It does not contain the precise competitor-employment exception.

The model correctly avoided claiming knowledge of an unseen contractual clause.

**Key lesson:**  
This is one of the strongest results in the experiment. The LLM did not use outside knowledge or fabricate provenance to compensate for retrieval failure.

---

## T05 — Passive investment threshold

**Query:**  
What percentage of a publicly traded competitor could Carter own without violating the agreement solely because of the investment?

**Top-1 retrieval:**  
DOC-2025-0142-02 — Employment Agreement - James Carter

**Cosine similarity:** `0.703075`

**Gold answer:**  
Less than 2%.

**LLM result:**  
Correct.

**Localization:**
- Page: `4`
- Section: `10.2`
- Evidence correctly identified the passive-investment clause.

**LLM classification:**
- Source sufficiency: `sufficient`
- Source role: `operative_source`

**Evaluation:** **Clean success and strong evidence localization**

The LLM successfully found a narrow clause inside the eight-page agreement.

**Schema note:**  
For this question, the Employment Agreement could reasonably be described as both a `primary_source` and an `operative_source`. The current enum requires one choice, so these categories overlap slightly.

---

## T06 — Customer non-solicitation duration

**Query:**  
How long was Carter prohibited from directly soliciting customers he had been responsible for before leaving Northstar?

**Top-1 retrieval:**  
DOC-2025-0142-09 — Settlement Agreement and Mutual Release

**Cosine similarity:** `0.785051`

**Preferred original-contract source rank:** `3`

**Gold benchmark answer:**  
Six months under the original Employment Agreement.

**LLM result:**  
The LLM answered from the Settlement Agreement that the restriction ran through March 21, 2026 and that the Settlement expressly characterized this as a six-month period from its effective date.

**LLM classification:**
- Source sufficiency: `sufficient`
- Source role: `operative_source`

**Evaluation:** **Correct for the literal retrieved-context question, but benchmark-confounded**

This result exposes an ambiguity in the frozen benchmark question.

The query does **not explicitly say**:

> "Under the original Employment Agreement..."

The Settlement Agreement independently imposes a customer restriction covering customers for whom Carter had material responsibility before leaving Northstar and explicitly supplies a six-month duration.

Therefore, given the literal query and the Top-1 Settlement, `sufficient` is defensible.

The LLM did not hallucinate. It answered a real restriction contained in the retrieved document.

**Key lesson:**  

> T06 does not cleanly distinguish the original contractual restriction from the later settlement restriction.

For consistency, the query should remain frozen through the current Top-1 / Top-2 / Oracle series, but this ambiguity must be noted when interpreting results.

A future benchmark version should use wording such as:

> "Under Carter's original Employment Agreement, how long was he prohibited from directly soliciting Restricted Customers after separation?"

---

## T07 — Restriction considered more defensible than the 50-mile non-compete

**Query:**  
What restriction was considered more defensible than preventing Carter from competing within fifty miles of Westbridge?

**Top-1 retrieval:**  
DOC-2025-0142-07 — Carter Opposition to Preliminary Injunction

**Cosine similarity:** `0.728537`

**Gold answer:**  
The six-month customer non-solicitation covenant.

**LLM result:**  
The model stated that the Opposition did not establish what restriction was *considered more defensible*. It only showed Carter requesting narrower relief involving confidential information and named-customer solicitation.

**LLM classification:**
- Source sufficiency: `insufficient`
- Source role: `advocacy`

**Evaluation:** **Excellent abstention**

The model correctly distinguished:

- a party proposing narrower relief, from
- an analytical conclusion that a particular restriction is more defensible.

**Key lesson:**  
Again, the LLM did not manufacture the gold answer when the retrieved document could not establish it.

---

## T08 — Relief Northstar requested from the court

**Query:**  
What did Northstar actually ask the court to stop Carter from doing at Vertex?

**Top-1 retrieval:**  
DOC-2025-0142-06 — Motion for Preliminary Injunction

**Cosine similarity:** `0.779587`

**Gold answer:**  
Northstar asked the court to stop Carter from providing substantially similar enterprise-sales services to Vertex within 50 miles of Westbridge for the remainder of the 12-month restriction and to enforce the six-month customer non-solicitation covenant.

**LLM result:**  
Correct.

**LLM classification:**
- Source sufficiency: `sufficient`
- Source role: `advocacy`

**Evaluation:** **Clean success**

The source is advocacy, but for a question asking **what Northstar asked the court to order**, Northstar's own Motion is directly sufficient.

**Key lesson:**  
`advocacy` does not mean `insufficient`. Source role must be interpreted relative to the question.

---

## T09 — Whether Carter could continue working for Vertex

**Query:**  
After the dispute was resolved, was Carter still allowed to work for Vertex?

**Top-1 retrieval:**  
DOC-2025-0142-09 — Settlement Agreement and Mutual Release

**Cosine similarity:** `0.744242`

**Gold answer:**  
Yes.

**LLM result:**  
Correct.

**Localization:**
- Page: `1`
- Section: `2. Employment at Vertex`

**LLM classification:**
- Source sufficiency: `sufficient`
- Source role: `operative_source`

**Evaluation:** **Clean success**

The Settlement is the operative instrument establishing the final negotiated outcome.

---

## T10 — Ultimate treatment of the broad geographic restriction

**Query:**  
What ultimately happened to the broad geographic restriction while Northstar continued protecting specific customer relationships?

**Top-1 retrieval:**  
DOC-2025-0142-10 — Matter Closing Memorandum

**Cosine similarity:** `0.751536`

**Preferred operative source:**  
Settlement Agreement (#2)

**Gold answer:**  
The dispute resolved without using the broad geographic restriction to bar Carter from Vertex. Northstar instead used targeted customer non-solicitation and confidentiality protections.

**LLM result:**  
Correct.

**LLM classification:**
- Source sufficiency: `sufficient`
- Source role: `internal_analysis`

**Evaluation:** **Correct and grounded**

Our original expectation of `partially_sufficient` was again too strict.

The Closing Memorandum is not the operative instrument creating the settlement terms, but it contains enough information to answer the retrospective question:

> "What ultimately happened?"

Therefore:

- `source_sufficiency = sufficient`
- `source_role = internal_analysis`

can both be true simultaneously.

**Key lesson:**  
Again, factual sufficiency and legal authority are different dimensions.

---

# 5. Summary Scorecard

| Query | Preferred source Top-1? | LLM outcome | Sufficiency | Source role | Hallucination / grounding assessment |
|---|---|---|---|---|---|
| T01 | No | Correct partial answer; abstained on missing duration | Partially sufficient | Advocacy | Strong |
| T02 | No | Correct | Sufficient | Advocacy | Strong |
| T03 | Yes | Correct | Sufficient | Internal analysis | Strong |
| T04 | No | Correctly abstained | Insufficient | Internal analysis | Excellent |
| T05 | Yes | Correct | Sufficient | Operative source | Strong |
| T06 | No (#3) | Correct for Settlement; benchmark ambiguous | Sufficient | Operative source | Strong |
| T07 | No | Correctly abstained | Insufficient | Advocacy | Excellent |
| T08 | Yes | Correct | Sufficient | Advocacy | Strong |
| T09 | Yes | Correct | Sufficient | Operative source | Strong |
| T10 | No | Correct | Sufficient | Internal analysis | Strong |

---

# 6. Aggregate Interpretation

## Retrieval performance remains the bottleneck

The whole-document retriever placed the preferred authoritative / operative source first for only:

**4 / 10 = 40%**

That weakness has not disappeared.

The LLM cannot recover an unseen authoritative source.

---

## LLM grounding behavior was very strong

Across the ten runs:

- No obvious unsupported citation to an unseen document was observed.
- T04 correctly abstained rather than inventing the missing contractual rule.
- T07 correctly abstained rather than converting Carter's requested narrower relief into an independent legal conclusion.
- T01 supplied only the portion actually supported by the Opposition and explicitly acknowledged that duration was missing.

This is more important than raw answer accuracy for a legal RAG system.

---

## Approximate answer-behavior breakdown

### Direct substantive answers
T02, T03, T05, T06, T08, T09, T10

**7 / 10**

### Correct abstentions caused by insufficient Top-1 context
T04, T07

**2 / 10**

### Correct partial answer with explicit missing evidence
T01

**1 / 10**

Thus all ten responses exhibited behavior that was broadly consistent with the supplied Top-1 evidence.

This does **not** mean the overall RAG system achieved 100% answer accuracy. It means the generative layer behaved responsibly given what retrieval supplied.

---

# 7. Major Finding: Sufficiency ≠ Authority

The biggest conceptual result from this experiment is:

> A document can be sufficient to answer a question without being the authoritative or operative legal source.

Examples:

### T02
The Opposition was advocacy, but contained enough evidence to explain why the 50-mile restriction was argued to be difficult to enforce.

### T08
The Motion was advocacy, but it was the direct source for what Northstar asked the court to do.

### T10
The Closing Memorandum was internal analysis / summary, but sufficiently described what ultimately happened.

Therefore the original assumption:

> "non-authoritative source → partially sufficient"

was incorrect.

The Pydantic schema's separation between:

- `source_sufficiency`
- `source_role`
- `authority_note`

proved useful.

---

# 8. Benchmark Issue Discovered: T06 Is Ambiguous

T06 was designed to test a failure where:

- Settlement ranked #1,
- Intake ranked #2,
- original Employment Agreement ranked #3.

However, the literal question can also be answered by the Settlement because the Settlement independently imposes a six-month customer restriction on substantially the same customer category.

Therefore the Top-1 LLM's `sufficient` classification is reasonable.

This means T06 is still useful for retrieval/provenance analysis, but it is not a perfectly isolated test of the original Employment Agreement.

For the current experimental series, keep T06 unchanged so Top-1, Top-2, and Oracle remain comparable.

For a future benchmark revision, explicitly anchor the question to the original Employment Agreement.

---

# 9. Schema Observation

The current `source_role` enum has a small overlap:

- `primary_source`
- `operative_source`

For example, the Employment Agreement is both:
- a primary source of the original contractual terms, and
- the operative agreement governing those original terms.

The Settlement is also both a primary source and the operative source for the final negotiated resolution.

For the current Top-1 / Top-2 / Oracle series, keep the schema unchanged to preserve comparability.

A future schema could split this into two dimensions, for example:

```text
source_nature:
- primary
- secondary
- advocacy
- internal_analysis

legal_status:
- operative
- non_operating
- unclear
```

That would remove the overlap.

---

# 10. What This Experiment Shows

Whole-document Top-1 retrieval is imperfect, but the generative LLM can often:

1. extract the right fact from a semantically relevant secondary document,
2. localize supporting evidence,
3. recognize source type,
4. distinguish a direct legal instrument from internal analysis or advocacy,
5. and abstain when the retrieved document lacks the needed fact.

However, the LLM cannot solve a source that was never retrieved.

That is the reason the next experiment matters.

---

# 11. Prediction for Top-2 → LLM

The prior retrieval ranks predict:

### Likely to improve materially with Top-2
- T01
- T02
- T04
- T07
- T10

because the preferred source is available at rank #2.

### Already strong at Top-1
- T03
- T05
- T08
- T09

These should remain stable.

### Special case
- T06

The original Employment Agreement is rank #3, so the Top-2 experiment will still exclude the intended original-contract source.

This makes T06 the most important negative control for Top-2.

---

# 12. Next Experiment

## Experiment 2B.1-B — Top-2 Whole Documents → LLM

The retrieval system will remain unchanged except that the LLM receives:

```text
Query
+
Top-1 document
+
Top-2 document
```

The LLM must then:

1. answer using only those two documents,
2. identify which document best supports each proposition,
3. distinguish conflicting or differently authoritative sources,
4. localize evidence,
5. and abstain if neither document is sufficient.

The exact same ten frozen queries should be reused.
