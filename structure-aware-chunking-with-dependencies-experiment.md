# Hamilton & Cole Legal RAG Retrieval Experiment

## Whole-Document vs Blind Fixed-Size Chunks vs Structure-Aware Chunks (V1)

**Matter:** `HC-2025-0142`  
**Corpus:** 10 Hamilton & Cole LLP / Carter–Northstar documents  
**Embedding model:** `gemini-embedding-2`  
**Similarity:** cosine similarity  
**Benchmark:** 10 frozen legal retrieval queries

---

## Why this experiment exists

This benchmark tests a problem that matters in legal RAG:

> The most semantically similar source is not always the preferred authoritative or operative source.

The same fact may appear in the original agreement, intake memo, legal analysis, motion, opposition, hearing notes, settlement, and closing memo. Dense retrieval can therefore find highly relevant text while still preferring the wrong source role.

Three strategies were tested:

1. **Whole-document retrieval**
2. **Blind fixed-size chunk retrieval**
3. **Structure-aware chunk retrieval V1**

Structure-aware V1 preserved legal sections and enriched each section's `retrieval_text` with dependency text from a legal-reference graph. The experiment is intentionally preserved because it exposed an architecture mistake: **unbounded transitive dependency expansion can make a focused clause embedding behave like a mini whole-document embedding.**

---

# Corpus

| ID | Document |
|---|---|
| `DOC-2025-0142-01` | Client Intake Memorandum |
| `DOC-2025-0142-02` | Employment Agreement - James Carter |
| `DOC-2025-0142-03` | Resignation Email - James Carter |
| `DOC-2025-0142-04` | Restrictive Covenant - Preliminary Legal Analysis |
| `DOC-2025-0142-05` | Demand Letter to James Carter |
| `DOC-2025-0142-06` | Motion for Preliminary Injunction |
| `DOC-2025-0142-07` | Carter Opposition to Preliminary Injunction |
| `DOC-2025-0142-08` | Preliminary Injunction Hearing Notes |
| `DOC-2025-0142-09` | Settlement Agreement and Mutual Release |
| `DOC-2025-0142-10` | Matter Closing Memorandum |

---

# Retrieval Strategies

## 1. Whole-document retrieval

Each canonical document was embedded as one unit.

```text
query
-> query embedding
-> cosine similarity against document embeddings
-> ranked documents
```

### Metrics

| Metric | Result |
|---|---:|
| Preferred-document Recall@1 | **40%** |
| Preferred-document Recall@2 | **100%** |
| MRR | **0.7000** |

Whole-document retrieval was strong as a candidate-document retriever, but long documents could have local evidence diluted by unrelated content.

## 2. Blind fixed-size chunk retrieval

```text
chunk size = 300 words
overlap    = 50 words
step       = 250 words
```

Chunk boundaries ignored legal structure.

### Metrics

| Metric | Result |
|---|---:|
| Preferred-document Recall@1 | **50%** |
| Preferred-document Recall@2 | **70%** |
| Preferred-document Recall@3 | **80%** |
| Preferred-document Recall@5 | **90%** |
| MRR | **0.6676** |

Blind chunking improved some locally specific queries but could separate a clause from definitions and cross-references required to interpret it.

## 3. Structure-aware chunks V1

```text
canonical document
-> legal structure parser
-> legal reference resolver
-> dependency graph
-> structure-aware chunks
-> retrieval_text enrichment
-> embeddings
-> cosine retrieval
```

Each chunk kept two representations:

```text
text
= exact local source text

retrieval_text
= local section
+ document/section context
+ dependency text
```

The design goal was to preserve exact source evidence while embedding a richer representation.

The V1 mistake was that `resolved_dependencies` effectively used unbounded transitive graph descendants. Some Section 12 chunks ended up with dependency sets such as:

```text
['1.2', '1.3', '1.4', '1.5', '1.6', '1.7',
 '2.1', '2.2', '5.3',
 '12.1', '12.3', '12.4', '12.5', '12.6', '12.7',
 '18', '19']
```

A supposedly focused section embedding could therefore contain a large fraction of the Employment Agreement.

---

# Frozen Benchmark Queries

1. **How long and how far did Carter's original non-compete restrict him?**
2. **Why was Carter's fifty-mile non-compete potentially difficult to enforce?**
3. **What did Judge Maren think about the fifty-mile geographic restriction?**
4. **Under the original employment agreement, did simply accepting employment with a competitor automatically violate the agreement?**
5. **What percentage of a publicly traded competitor could Carter own without violating the agreement solely because of the investment?**
6. **How long was Carter prohibited from directly soliciting customers he had been responsible for before leaving Northstar?**
7. **What restriction was considered more defensible than preventing Carter from competing within fifty miles of Westbridge?**
8. **What did Northstar actually ask the court to stop Carter from doing at Vertex?**
9. **After the dispute was resolved, was Carter still allowed to work for Vertex?**
10. **What ultimately happened to the broad geographic restriction while Northstar continued protecting specific customer relationships?**

---

# Main Comparison

> **Preferred-source rank** means the first rank at which the preferred document appears.  
> For structure-aware retrieval, this is not always the same as the rank of the correct answer-bearing clause.

| Test | Preferred source | Whole-doc preferred rank | Blind preferred rank | Structure-aware V1 preferred rank | Whole-doc Top-1 | Blind Top-1 | Structure-aware V1 Top-1 |
|---|---|---:|---:|---:|---|---|---|
| T01 | Employment Agreement (`DOC-02`) | 2 | 2 | **2** | Carter Opposition | Carter Opposition | Top-1 not preserved in final capture |
| T02 | Preliminary Legal Analysis (`DOC-04`) | 2 | 2 | **Not preserved** | Carter Opposition | Carter Opposition | Structure-aware output not preserved |
| T03 | Hearing Notes (`DOC-08`) | 1 | 1 | **Not fully preserved** | Hearing Notes | Hearing Notes | Final capture begins mid-result |
| T04 | Employment Agreement (`DOC-02`) | 2 | 1 | **1** | Client Intake Memo | Employment Agreement chunk | Employment Agreement §12.4 |
| T05 | Employment Agreement (`DOC-02`) | 1 | 1 | **1** | Employment Agreement | Employment Agreement chunk | Employment Agreement §10.2 |
| T06 | Employment Agreement (`DOC-02`) | 2 | **7** | **2*** | Settlement Agreement | Settlement Agreement chunk | Settlement Agreement §3 |
| T07 | Preliminary Legal Analysis (`DOC-04`) | 2 | 3 | **5** | Carter Opposition | Carter Opposition | Carter Opposition |
| T08 | Motion for Preliminary Injunction (`DOC-06`) | 1 | 1 | **2** | Motion | Motion | Carter Opposition |
| T09 | Settlement Agreement (`DOC-09`) | 1 | 1 | **1** | Settlement Agreement | Settlement Agreement | Settlement Agreement §2 |
| T10 | Settlement Agreement (`DOC-09`) | 2 | **5** | **2** | Closing Memorandum | Closing Memorandum | Closing Memorandum |

\* **T06 needs special treatment:** the Employment Agreement first appears at rank 2 through **§12.6**, which is an employee-nonsolicitation clause and is not the answer-bearing customer-nonsolicitation provision. The actually relevant **§12.5** appears at **rank 4**.

---

# Key Query Deep Dives

## T04 - Chunking localizes an explicit contract rule

**Query**

> Under the original employment agreement, did simply accepting employment with a competitor automatically violate the agreement?

### Whole document

```text
Rank 1: Client Intake Memorandum
Rank 2: Employment Agreement
```

### Blind chunks

```text
Rank 1: Employment Agreement chunk
```

### Structure-aware V1

```text
Rank 1: Employment Agreement §12.4
Score: 0.750118
```

The exact clause begins:

```text
Nothing in this Agreement automatically prohibits Employee from accepting
employment with a Competitor.
```

This is a clean example of local chunk retrieval outperforming the whole-document embedding.

---

## T05 - Strong local clause, easy retrieval

**Query**

> What percentage of a publicly traded competitor could Carter own without violating the agreement solely because of the investment?

### Structure-aware V1

```text
Rank 1: Employment Agreement §10.2
Score: 0.813273
```

The answer is highly local and does not require a large dependency chain, so the structured clause performs cleanly.

---

# T06 - The Most Important Result

**Query**

> How long was Carter prohibited from directly soliciting customers he had been responsible for before leaving Northstar?

**Preferred source:** Employment Agreement (`DOC-02`)  
**Correct answer-bearing clause:** §12.5, interpreted with §1.4 and §1.5

## Whole-document retrieval

```text
Rank 1: Settlement Agreement
Rank 2: Employment Agreement
```

**Preferred-document rank: 2**

The whole Employment Agreement vector aggregates signals from §12.5, §1.4, §1.5, duration language, post-employment restrictions, and the broader Carter/Northstar context.

## Blind chunk retrieval

```text
Rank 1: Settlement Agreement
Rank 2: Client Intake Memorandum
Rank 3: Carter Opposition
Rank 4: Motion for Preliminary Injunction
Rank 5: Closing Memorandum
...
Rank 7: first Employment Agreement chunk
```

**Preferred-document rank: 7**

The relationship needed for the query is:

```text
§12.5
  -> "Restricted Customer"
      -> §1.4
          -> "Material Responsibility"
              -> §1.5
```

Blind windows split this legal relationship apart, while short secondary documents summarized it locally.

## Structure-aware V1

```text
Rank 1
Settlement Agreement §3
Score: 0.842057

Rank 2
Employment Agreement §12.6
Score: 0.821072

Rank 3
Client Intake Memorandum
Score: 0.804625

Rank 4
Employment Agreement §12.5
Score: 0.804154
```

At the **document level**, the Employment Agreement improves dramatically:

```text
Blind:      rank 7
Structured: rank 2
```

But that headline hides a clause-level failure.

The rank-2 clause is §12.6, which concerns solicitation of **employees**, not customers:

```text
For six (6) months after separation, Employee will not directly solicit a Company
employee whom Employee supervised...
```

The actual customer nonsolicitation clause is §12.5 at **rank 4**:

```text
For six (6) months after separation, Employee will not directly solicit a
Restricted Customer, as defined in Section 1.4...
```

This is one of the most important lessons from the benchmark:

```text
preferred document rank
!=
preferred answer-bearing clause rank
```

The settlement still wins because it expresses the query almost literally and locally: direct solicitation, Northstar customers, material responsibility, and the six-month period all appear together.

Dense semantic retrieval is therefore doing something reasonable while still missing the preferred authority for an **original-contract** question.

---

## T07 - More structural context is not automatically better

**Query**

> What restriction was considered more defensible than preventing Carter from competing within fifty miles of Westbridge?

**Preferred source:** Preliminary Legal Analysis (`DOC-04`)

```text
Whole-document preferred rank: 2
Blind preferred rank:          3
Structure-aware preferred rank: 5
```

Structure-aware V1 Top-1:

```text
Carter Opposition to Preliminary Injunction
Score: 0.734173
```

The preferred Legal Analysis Memo fell to rank 5.

This is evidence that blindly injecting more structural context can hurt ranking rather than help it.

---

## T08 - Authority and semantic similarity separate again

**Query**

> What did Northstar actually ask the court to stop Carter from doing at Vertex?

**Preferred source:** Motion for Preliminary Injunction (`DOC-06`)

### Whole document

```text
Rank 1: Motion for Preliminary Injunction
```

### Blind chunks

```text
Rank 1: Motion for Preliminary Injunction
```

### Structure-aware V1

```text
Rank 1: Carter Opposition
Score: 0.794559

Rank 2: Motion for Preliminary Injunction
Score: 0.790091
```

The score gap is only:

```text
0.004468
```

The Opposition is semantically close because it discusses the same injunction dispute. But the query asks what **Northstar requested**, making the Motion the preferred source role.

Again:

```text
semantic relevance
!=
source role
!=
authority
```

---

## T09 - Clean success

**Query**

> After the dispute was resolved, was Carter still allowed to work for Vertex?

All three approaches put the Settlement Agreement first.

Structure-aware V1:

```text
Rank 1: Settlement Agreement §2
Score: 0.773480
```

This is a relatively easy retrieval problem because the final operative document directly answers the question.

---

## T10 - Structure recovers the preferred source earlier than blind chunks

**Query**

> What ultimately happened to the broad geographic restriction while Northstar continued protecting specific customer relationships?

**Preferred source:** Settlement Agreement (`DOC-09`)

```text
Whole document:
Rank 1 Closing Memorandum
Rank 2 Settlement Agreement

Blind:
Rank 1 Closing Memorandum
Rank 5 Settlement Agreement

Structure-aware V1:
Rank 1 Closing Memorandum (0.759470)
Rank 2 Settlement Agreement §3 (0.738445)
```

Structure-aware retrieval restores the preferred source from blind rank 5 to rank 2.

The Closing Memorandum still wins because it compresses the final outcome into highly concentrated semantic language.

---

# What Structure-Aware V1 Actually Taught Us

The experiment did **not** show that structure-aware chunking is useless.

It showed that the first dependency-expansion policy was too broad.

The intended representation was:

```text
primary clause
+ useful structural context
+ necessary referenced definitions
```

But some chunks became:

```text
primary clause
+ direct references
+ every transitively reachable reference
+ a large fraction of the contract
```

That creates **dependency explosion**.

For example, §12.2 directly depended on:

```text
['1.6', '12.1']
```

but its resolved set expanded to roughly seventeen sections.

Instead of embedding:

```text
§12.2
+ §12.1
+ §1.6
```

the vector represented something closer to:

```text
§12.2
+ a large fraction of the Employment Agreement
```

This partially recreates whole-document retrieval inside each chunk.

---

# Core Lessons

## 1. Retrieval relevance is not source authority

Dense cosine similarity primarily asks:

```text
Which embedding is semantically closest to this query?
```

It does not inherently ask:

```text
Which document is the legally preferred source for this question?
```

Source role depends on query intent:

```text
original contract term -> Employment Agreement
party's requested relief -> Motion
judge's observation -> Hearing Notes
final negotiated obligation -> Settlement Agreement
```

## 2. Source sufficiency is different from authority

A Closing Memorandum may contain enough information to answer a question, but the Settlement Agreement may still be the preferred operative source.

An Opposition may accurately describe an injunction dispute, while the Motion remains the preferred source for what Northstar actually requested.

## 3. Blind chunking can destroy distributed legal meaning

Legal provisions often depend on:

- defined terms,
- cross-references,
- parent clauses,
- duration provisions,
- exceptions,
- schedules.

A fixed 300-word window does not know those relationships.

## 4. Unlimited graph traversal is also wrong

A dependency graph tells us where the system **can navigate**.

It should not automatically mean:

```text
put every reachable section in every embedding
```

These are separate:

```text
graph reachability
!=
embedding context
!=
LLM answer context
```

## 5. Document-level Recall can hide clause-level failure

T06 demonstrates this.

```text
Structure-aware preferred document rank = 2
```

looks strong.

But:

```text
rank 2 = §12.6 employee nonsolicitation
rank 4 = §12.5 customer nonsolicitation
```

So legal-RAG evaluation should eventually track both:

```text
preferred document rank
and
preferred / answer-bearing clause rank
```

---

# Planned Structure-Aware V2

Keep the reference graph, but use **controlled expansion**:

```text
primary clause
+
document / parent structural context
+
direct dependencies
+
necessary definition-chain dependencies
+
strict depth / section / token budget
```

Example:

```text
§12.5
  -> §1.4 Restricted Customer
      -> §1.5 Material Responsibility
```

Those three sections may belong together.

But:

```text
§12.2
-> §12.1
-> dozens of transitively connected clauses
```

should not recursively absorb the whole contract.

A stronger production architecture is:

```text
query
-> authorization / metadata filtering
-> dense + lexical retrieval
-> candidate chunks
-> reranking
-> bounded cross-reference expansion
-> source-role / authority logic
-> grounded LLM answer
-> citations
```

---

# Experiment Progression

```text
Whole-document retrieval
    |
    | strong candidate-document recall
    | weak local precision / long-document dilution
    v

Blind fixed-size chunks
    |
    | better locality on some queries
    | breaks cross-reference and definition relationships
    | major T06 degradation
    v

Structure-aware chunks V1
    |
    | restores legal relationships
    | improves several contract-local queries
    | recovers preferred sources earlier on T06 / T10
    | BUT expands transitive dependencies too aggressively
    v

Structure-aware chunks V2
    |
    | bounded dependency expansion
    | definition-aware traversal
    | preserve locality
    | dynamic context expansion after retrieval
    v

Authority-aware legal retrieval
```

The biggest lesson is that a robust legal RAG system must distinguish:

```text
semantic relevance
source sufficiency
source authority
clause-level evidence
cross-reference dependencies
```

Those concerns should not all be pushed into one embedding.

---

# Data Completeness Note

The final structure-aware terminal capture used for this report contains complete result blocks for **T04-T10**.

For **T01**, the preserved run confirms Employment Agreement §12.2 at **rank 2**, but the final Top-1 row was not retained in the capture.

For **T02**, the final structure-aware result block was not preserved in the supplied terminal output.

For **T03**, the supplied capture begins partway through the result block, so its complete Top-1/Top-2 output is not retained.

Those cells are intentionally marked rather than reconstructed from assumptions.

The whole-document and blind-chunk results are complete.

---

# Reproducibility Note

The benchmark uses the same frozen query wording and the same preferred-source labels across retrieval strategies.

When Structure-Aware V2 is implemented, the same 10 queries should be rerun **without changing their wording** so the new architecture can be compared directly with this report.
