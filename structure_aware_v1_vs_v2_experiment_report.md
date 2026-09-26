# Hamilton & Cole Retrieval Experiments — Final Comparison

## Purpose

This report concludes the first four dense-retrieval experiments performed on the synthetic **Hamilton & Cole LLP / Northstar Analytics** matter (`HC-2025-0142`).

The goal of these experiments was not simply to maximize a retrieval score. The goal was to understand, step by step, how different document representations affect semantic retrieval in a legal RAG system.

The four experiments were:

1. **Whole-document dense retrieval**
2. **Blind fixed-size chunk retrieval**
3. **Structure-aware retrieval V1 — full resolved-dependency enrichment**
4. **Structure-aware retrieval V2 — definition-only resolved-dependency enrichment**

Across all four experiments, the benchmark queries, preferred sources, embedding model, and cosine-similarity retrieval logic were kept fixed as much as possible so that changes in retrieval behavior could be attributed to the representation strategy itself.

---

# 1. Corpus

Matter:

`HC-2025-0142`

Documents:

1. `DOC-2025-0142-01` — Client Intake Memorandum
2. `DOC-2025-0142-02` — Employment Agreement - James Carter
3. `DOC-2025-0142-03` — Resignation Email
4. `DOC-2025-0142-04` — Restrictive Covenant - Preliminary Legal Analysis
5. `DOC-2025-0142-05` — Demand Letter
6. `DOC-2025-0142-06` — Motion for Preliminary Injunction
7. `DOC-2025-0142-07` — Carter Opposition to Preliminary Injunction
8. `DOC-2025-0142-08` — Preliminary Injunction Hearing Notes
9. `DOC-2025-0142-09` — Settlement Agreement and Mutual Release
10. `DOC-2025-0142-10` — Matter Closing Memorandum

Embedding model:

`gemini-embedding-2`

Embedding dimension:

`3072`

Retrieval method:

`query embedding -> cosine similarity -> descending rank`

---

# 2. Frozen Benchmark Queries

| ID | Query | Preferred Source |
|---|---|---|
| T01 | How long and how far did Carter's original non-compete restrict him? | DOC-02 — Employment Agreement |
| T02 | Why was Carter's fifty-mile non-compete potentially difficult to enforce? | DOC-04 — Preliminary Legal Analysis |
| T03 | What did Judge Maren think about the fifty-mile geographic restriction? | DOC-08 — Hearing Notes |
| T04 | Under the original employment agreement, did simply accepting employment with a competitor automatically violate the agreement? | DOC-02 — Employment Agreement |
| T05 | What percentage of a publicly traded competitor could Carter own without violating the agreement solely because of the investment? | DOC-02 — Employment Agreement |
| T06 | How long was Carter prohibited from directly soliciting customers he had been responsible for before leaving Northstar? | DOC-02 — Employment Agreement |
| T07 | What restriction was considered more defensible than preventing Carter from competing within fifty miles of Westbridge? | DOC-04 — Preliminary Legal Analysis |
| T08 | What did Northstar actually ask the court to stop Carter from doing at Vertex? | DOC-06 — Motion for Preliminary Injunction |
| T09 | After the dispute was resolved, was Carter still allowed to work for Vertex? | DOC-09 — Settlement Agreement |
| T10 | What ultimately happened to the broad geographic restriction while Northstar continued protecting specific customer relationships? | DOC-09 — Settlement Agreement |

The benchmark intentionally mixes several legal information needs:

- original contract terms,
- enforceability analysis,
- judicial observations,
- requested relief,
- negotiated settlement terms,
- final matter outcome.

This matters because the most semantically similar document is not always the legally preferred source.

---

# 3. Final Preferred-Source Rank Comparison

The table below uses the rank of the preferred **document**.

For T06, a separate clause-level note is necessary because both structure-aware systems surfaced the Employment Agreement at rank 2 through the wrong clause (`§12.6`) while the correct answer-bearing clause (`§12.5`) remained rank 4.

| Query | Whole Document | Blind Chunks | Structure V1 | Structure V2 |
|---|---:|---:|---:|---:|
| T01 | 2 | 2 | 2 | **1** |
| T02 | 2 | 2 | 3 | 3 |
| T03 | **1** | **1** | **1** | **1** |
| T04 | 2 | **1** | **1** | **1** |
| T05 | **1** | **1** | **1** | **1** |
| T06 | 2 | 7 | 2* | 2* |
| T07 | 2 | 3 | 5 | **4** |
| T08 | **1** | **1** | 2 | 2 |
| T09 | **1** | **1** | **1** | **1** |
| T10 | 2 | 5 | 2 | 2 |

`*` T06 document-level rank is misleading. In both Structure V1 and Structure V2, the first Employment Agreement result was `§12.6` at rank 2, while the correct customer non-solicitation clause `§12.5` was rank 4.

---

# 4. Aggregate Metrics

These metrics use the preferred-document rank shown above.

| Strategy | Recall@1 | Recall@2 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|---:|
| Whole document | 40% | **100%** | 100% | 100% | **0.7000** |
| Blind fixed-size chunks | **50%** | 70% | 80% | 90% | 0.6676 |
| Structure-aware V1 | 40% | 80% | 90% | **100%** | 0.6533 |
| Structure-aware V2 | **50%** | 80% | 90% | **100%** | **0.7083** |

Important caveat:

These aggregate metrics measure preferred **document** retrieval, not answer-bearing **section** retrieval.

That distinction is especially important for T06.

---

# 5. Experiment 1 — Whole-Document Dense Retrieval

## Representation

Each complete document was embedded as one vector.

Conceptually:

```text
complete document text
        ↓
Gemini embedding
        ↓
one vector per document
```

## Result

Whole-document retrieval performed surprisingly well on this small corpus.

Preferred-source ranks:

```text
T01  2
T02  2
T03  1
T04  2
T05  1
T06  2
T07  2
T08  1
T09  1
T10  2
```

Metrics:

```text
Recall@1 = 40%
Recall@2 = 100%
MRR      = 0.7000
```

## What Worked

Whole-document embeddings were effective when the relevant legal meaning was distributed across several parts of a document.

The Employment Agreement is the clearest example.

A single contract vector could simultaneously absorb semantic signals from:

- definitions,
- restrictive-covenant provisions,
- duration clauses,
- geographic terms,
- competitor restrictions,
- customer non-solicitation language.

This gave long authoritative documents a useful aggregate semantic representation.

## What Failed

The same aggregation that helped completeness also reduced precision.

A whole Employment Agreement vector does not identify which clause contains the answer.

It only says:

> this document, as a whole, is semantically related to the query.

Whole-document retrieval therefore has weak evidence localization.

It can also be expensive later because an answer model may receive far more text than it needs.

## Key Lesson

Whole-document retrieval can be a strong baseline because aggregation preserves distributed meaning.

However:

```text
good document retrieval
≠
good evidence localization
```

---

# 6. Experiment 2 — Blind Fixed-Size Chunking

## Representation

Documents were divided using fixed word windows:

```text
chunk size = 300 words
overlap    = 50 words
step       = 250 words
```

No attempt was made to respect:

- clause boundaries,
- headings,
- legal definitions,
- cross-references,
- schedules,
- semantic units.

## Result

Preferred ranks:

```text
T01  2
T02  2
T03  1
T04  1
T05  1
T06  7
T07  3
T08  1
T09  1
T10  5
```

Metrics:

```text
Recall@1 = 50%
Recall@2 = 70%
Recall@3 = 80%
Recall@5 = 90%
MRR      = 0.6676
```

## What Improved

Blind chunking increased local precision for some questions.

T04 and T05 benefited because the answer-bearing contract language could appear in relatively focused chunks.

This improved Recall@1 from 40% to 50%.

## Major Failure — T06

T06 asked:

> How long was Carter prohibited from directly soliciting customers he had been responsible for before leaving Northstar?

The correct answer depends on a legal relationship:

```text
§12.5 Customer Non-Solicitation
        ↓
§1.4 Restricted Customer
        ↓
§1.5 Material Responsibility
```

Blind chunking broke this relationship apart.

The answer-bearing Employment Agreement content fell to rank 7.

Meanwhile shorter documents such as:

- the Settlement Agreement,
- Intake Memorandum,
- Opposition,
- Motion,
- Closing Memorandum,

contained compact natural-language summaries of the same dispute.

Those summaries were easier for dense semantic retrieval to match.

## Why This Happens

Dense retrieval independently compares:

```text
query vector
vs.
chunk vector
```

It does not automatically understand:

```text
§12.5 refers to §1.4
and
§1.4 depends on §1.5
```

Blind chunks therefore penalize legal drafting styles where meaning is intentionally distributed across defined terms and cross-references.

## Key Lesson

Smaller chunks can improve semantic precision, but arbitrary boundaries can destroy legal meaning.

```text
smaller chunk
≠
better legal chunk
```

---

# 7. Experiment 3 — Structure-Aware V1

## Goal

V1 attempted to preserve legal structure and reconnect meaning that blind chunking had separated.

Pipeline:

```text
canonical document
        ↓
legal structure parser
        ↓
sections / clauses
        ↓
reference detection
        ↓
dependency graph
        ↓
resolved dependencies
        ↓
context-enriched retrieval_text
        ↓
embedding
```

Each chunk kept two different text representations:

```text
text
= exact local legal source text

retrieval_text
= semantic text used for embedding
```

## V1 Retrieval Representation

V1 embedded:

```text
primary section
+
ALL resolved dependency text
```

For example, a clause such as `§12.5` could inherit a large transitive dependency set.

The graph itself was useful, but all graph reachability was treated as embedding context.

## Result

Preferred ranks:

```text
T01  2
T02  3
T03  1
T04  1
T05  1
T06  2 document-level / 4 correct clause
T07  5
T08  2
T09  1
T10  2
```

Metrics:

```text
Recall@1 = 40%
Recall@2 = 80%
Recall@3 = 90%
Recall@5 = 100%
MRR      = 0.6533
```

## What Worked

Structure-aware parsing produced much better evidence units.

Examples:

- T04 retrieved `§12.4` directly.
- T05 retrieved `§10.2` directly.
- T09 retrieved Settlement `§2`.
- individual clauses preserved section IDs and page provenance.

This is a major architectural improvement even when the preferred-document metric does not move.

The system now knows:

```text
document
section
page
exact text
references
dependencies
```

instead of only knowing an arbitrary 300-word window.

## What Failed

The dependency graph became too powerful for the embedding layer.

Example:

```text
§12.2 direct dependencies:
['1.6', '12.1']

§12.2 resolved dependencies:
['1.2', '1.3', '1.4', '1.5', '1.6', '1.7',
 '2.1', '2.2', '5.3', '12.1', '12.3', '12.4',
 '12.5', '12.6', '12.7', '18', '19']
```

Embedding all of those sections caused a local clause vector to become semantically similar to a miniature Employment Agreement vector.

This recreated part of the original whole-document problem.

## Core V1 Discovery

The most important conceptual discovery was:

```text
graph reachability
≠
embedding context
≠
LLM answer context
```

A section being reachable in a dependency graph does not mean its full text should permanently affect the primary section's vector.

## T06 Still Exposed a Clause-Level Failure

For T06:

```text
Rank 1  Settlement §3
Rank 2  Employment Agreement §12.6
Rank 3  Intake Memo
Rank 4  Employment Agreement §12.5
```

The preferred document appeared at rank 2, but the first contract result was the wrong clause.

Therefore a document-level benchmark alone would incorrectly suggest that retrieval had solved T06.

This led to an important evaluation lesson:

```text
preferred document rank
and
answer-bearing clause rank

should both be measured
```

---

# 8. Experiment 4 — Structure-Aware V2

## Goal

V2 preserved the exact same:

- parser,
- reference resolver,
- dependency graph,
- embedding model,
- cosine retrieval,
- benchmark queries.

Only the retrieval representation changed.

This made V2 a controlled experiment.

## V2 Retrieval Representation

Instead of embedding every resolved dependency, V2 filtered the resolved dependency set and embedded only sections classified as definitions.

Conceptually:

```text
resolved dependencies
        ↓
is this section a definition?
       / \
     yes  no
      ↓    ↓
   embed  graph only
```

The chunk also stored:

```text
embedded_definition_section_ids
```

This explicitly separates:

```text
what the graph knows

from

what actually affected the embedding
```

## Result

Preferred ranks:

```text
T01  1
T02  3
T03  1
T04  1
T05  1
T06  2 document-level / 4 correct clause
T07  4
T08  2
T09  1
T10  2
```

Metrics:

```text
Recall@1 = 50%
Recall@2 = 80%
Recall@3 = 90%
Recall@4 = 100%
Recall@5 = 100%
MRR      = 0.7083
```

V2 produced the highest MRR of the four tested representations on this benchmark, although the corpus is too small to treat the difference as a general production claim.

---

# 9. V1 vs V2 — What Actually Changed?

The difference is not:

```text
V1 embeds direct dependencies
V2 embeds only resolved definitions
```

That would be inaccurate.

The actual difference is:

## V1

```text
primary section
+
every section in resolved_dependencies
```

## V2

```text
primary section
+
only definition sections found inside resolved_dependencies
```

`direct_dependencies` remained useful graph metadata in both versions.

V2 did not separately embed every direct dependency.

---

# 10. Evidence That V2 Reduced Semantic Contamination

T07 is a useful example.

Query:

> What restriction was considered more defensible than preventing Carter from competing within fifty miles of Westbridge?

Preferred source:

`DOC-04 — Preliminary Legal Analysis`

### V1

```text
Rank 1  Opposition
Rank 2  Employment Agreement §1.6
Rank 3  Employment Agreement §12.2
Rank 4  Employment Agreement §12.1
Rank 5  Legal Analysis
```

### V2

```text
Rank 1  Opposition
Rank 2  Employment Agreement §12.2
Rank 3  Employment Agreement §1.6
Rank 4  Legal Analysis
Rank 5  Employment Agreement §12.1
```

The Legal Analysis itself did not suddenly become more semantically similar.

Its score remained:

`0.711198`

Instead, `§12.1` became less artificially broad and dropped below it.

This is an important result.

V2 can help not only by increasing the score of a good chunk, but by decreasing the score of chunks that previously matched too many unrelated concepts because of excessive dependency enrichment.

A useful description is:

> V2 reduced semantic contamination in some contract-section embeddings.

---

# 11. Controlled-Experiment Evidence

Several chunks with unchanged retrieval text produced exactly the same cosine scores across V1 and V2.

Examples:

## T08

```text
Opposition
V1 = 0.794559
V2 = 0.794559

Motion
V1 = 0.790091
V2 = 0.790091
```

## T09

```text
Settlement §2
V1 = 0.773480
V2 = 0.773480
```

## T10

```text
Closing Memo
V1 = 0.759470
V2 = 0.759470

Settlement §3
V1 = 0.738445
V2 = 0.738445
```

This is valuable because it confirms that the experiment was genuinely controlled.

Where the representation did not change, the semantic relationship did not change.

Where the representation changed, ranking differences can be attributed to the new retrieval text rather than a new embedding model or retrieval algorithm.

---

# 12. T10 and the Relevance-vs-Authority Problem

T10 asks:

> What ultimately happened to the broad geographic restriction while Northstar continued protecting specific customer relationships?

V2 returned:

```text
Rank 1  Matter Closing Memorandum
Rank 2  Settlement Agreement §3
```

The preferred source is the Settlement Agreement because it is the operative resolution.

However, the Closing Memorandum is a very strong semantic summary of the exact outcome.

This demonstrates a limitation that structure-aware chunking alone cannot solve.

Dense similarity mainly asks:

> Which text is semantically closest to this query?

It does not inherently ask:

> Which source is legally authoritative for this type of question?

Those are different objectives.

For legal RAG, retrieval quality therefore has at least three dimensions:

1. **Semantic relevance** — is the text about the question?
2. **Source sufficiency** — does the text contain enough information to answer?
3. **Authority / provenance** — is this the preferred legal source for the question?

T10 is a case where a secondary summary can be semantically excellent while an operative agreement is legally preferable.

---

# 13. T06 After V2

T06 remained the most important unresolved case.

V2 results:

```text
Rank 1  Settlement Agreement §3        0.842057
Rank 2  Employment Agreement §12.6    0.821072
Rank 3  Client Intake Memorandum       0.804625
Rank 4  Employment Agreement §12.5    0.804154
Rank 5  Motion for Preliminary Injunction
```

Correct answer-bearing source:

`Employment Agreement §12.5`

V2 correctly enriched `§12.5` with:

```text
embedded definitions:
['1.2', '1.4', '1.5']
```

The relevant legal chain is:

```text
§12.5
  ↓
§1.4 Restricted Customer
  ↓
§1.5 Material Responsibility
```

However, V2 still did not move `§12.5` above the semantically simpler summaries.

This demonstrates that definition enrichment alone cannot guarantee perfect dense ranking.

The Settlement Agreement says almost exactly:

```text
directly solicit
customer
material responsibility
six-month period
```

in one compact local passage.

That is naturally very attractive to a dense embedding model.

---

# 14. New V2 Failure Class — Definition Explosion

V2 solved a major part of V1's dependency explosion, but it exposed another issue.

Example:

```text
§12.2 direct dependencies:
['1.6', '12.1']
```

Yet V2 embedded:

```text
['1.2', '1.3', '1.4', '1.5', '1.6', '1.7']
```

Why?

Because the algorithm currently performs:

```text
all resolved dependencies
        ↓
keep every reachable definition
```

That is narrower than V1, but still broader than necessary.

This can be described as:

> definition explosion

The clause directly needs `§1.6`, but the transitive graph makes many other definitions reachable.

---

# 15. Planned V3

V3 should no longer ask:

> Which definitions exist anywhere inside the resolved dependency closure?

Instead it should ask:

> Which defined terms does this primary clause actually use?

Then it should follow only definition-to-definition edges.

Example:

```text
§12.5
  ↓ actual defined-term reference
§1.4 Restricted Customer
  ↓ actual defined-term reference
§1.5 Material Responsibility
```

The target representation becomes:

```text
retrieval_text =
§12.5
+
§1.4
+
§1.5
```

rather than every definition reachable through unrelated graph paths.

V3 should use bounded deterministic traversal with guardrails such as:

```text
definition-only edges
max traversal depth
max embedded definition count
token / character budget
cycle detection
```

This keeps ingestion deterministic and query-independent.

No LLM is required at embedding time.

---

# 16. Ingestion-Time Context vs Query-Time Context

A major architecture lesson from these experiments is that retrieval representation and final answer context should not be treated as the same thing.

## Ingestion Time

The system creates a compact, stable retrieval representation:

```text
document
↓
parse legal structure
↓
build typed reference graph
↓
select bounded definition context
↓
retrieval_text
↓
embedding
↓
vector store
```

## Query Time

After retrieval, the system can use the graph more dynamically:

```text
query
↓
dense / hybrid retrieval
↓
candidate chunks
↓
reranking
↓
primary evidence chunk
↓
inspect graph
↓
bounded query-aware context expansion
↓
LLM
```

For example:

```text
retrieve §12.5
↓
fetch §1.4
↓
fetch §1.5
↓
possibly fetch §12.3 if the query requires duration context
↓
stop
```

The key distinction is:

```text
embedding context
≠
final LLM evidence context
```

---

# 17. What These Experiments Say About Dense Retrieval

Dense retrieval is good at semantic paraphrase.

Examples:

```text
"customers he had been responsible for"
```

can semantically match concepts such as:

```text
Restricted Customer
Material Responsibility
```

However, dense retrieval does not inherently understand legal source hierarchy or document authority.

It also does not automatically reconstruct cross-reference chains unless that meaning is represented in the embedding text.

At the same time, adding more context is not monotonically better.

Too little context:

```text
blind chunk
→ meaning fragmented
```

Too much context:

```text
V1
→ semantic dilution / contamination
```

The objective is therefore not:

> maximize retrieval text

It is:

> create the smallest representation that preserves the semantic meaning needed for retrieval.

This is the central representation lesson from the experiment series.

---

# 18. Representation Progression

The four experiments can be summarized as follows.

## Whole Document

```text
too broad
but semantically complete
```

Strength:

- preserves distributed meaning.

Weakness:

- poor evidence localization.

---

## Blind Fixed-Size Chunking

```text
more local
but structurally unaware
```

Strength:

- improves precision for self-contained passages.

Weakness:

- breaks definitions and cross-references.

---

## Structure-Aware V1

```text
correct legal units
but too much graph context
```

Strength:

- restores legal structure and provenance.

Weakness:

- transitive dependency expansion creates mini whole-document vectors.

---

## Structure-Aware V2

```text
correct legal units
+
definition-only enrichment
```

Strength:

- reduces semantic contamination.
- highest MRR in this small benchmark.
- preserves full graph separately from embedding context.

Weakness:

- all reachable definitions may still be too broad.
- does not solve source authority.
- T06 correct clause still ranks 4.

---

# 19. Final Conclusions

The experiments show that chunking is not merely a text-splitting problem.

For legal RAG, chunking is a **representation problem**.

The system must decide:

```text
What is the primary evidence unit?

What context is required to understand it?

What information belongs in the vector?

What relationships should remain only in metadata / graph form?

What evidence should be fetched dynamically after retrieval?
```

The experiments also show that no single dense-retrieval representation should be expected to solve every retrieval failure.

Some failures come from:

- poor chunk boundaries,
- missing definitions,
- semantic dilution,
- lexical specificity,
- competing summaries,
- document authority,
- temporal status,
- query intent.

These require different retrieval layers.

---

# 20. Next Architecture Direction

After the bounded definition-chain experiment, the retrieval architecture should progressively move toward:

```text
query
↓
authorization / metadata filtering
↓
dense retrieval
+
lexical retrieval / BM25
↓
candidate merge
↓
reranking
↓
bounded graph expansion
↓
source-role / authority logic
↓
grounded LLM answer
↓
citations
```

Each addition should be evaluated independently against the frozen benchmark.

This preserves the experimental discipline established in the first four retrieval experiments.

---

# 21. Main Engineering Lessons

1. **Whole-document vectors can outperform naive chunks when meaning is distributed across a legal document.**

2. **Blind chunking can destroy legal semantics even when overlap is used.**

3. **Cross-reference graphs are useful, but graph reachability should not automatically become embedding context.**

4. **Structure-aware chunks improve evidence localization even when aggregate retrieval metrics do not dramatically improve.**

5. **Definitions are especially valuable retrieval context because legal clauses often depend on them for semantic meaning.**

6. **Embedding every reachable definition is still too broad; definition traversal should be typed and bounded.**

7. **Document-level evaluation can hide clause-level retrieval failures.**

8. **Semantic relevance, answer sufficiency, and legal authority are separate retrieval objectives.**

9. **A short secondary document may outrank an authoritative primary document because it summarizes the exact query semantics more compactly.**

10. **Embedding context should remain compact; richer context can be assembled dynamically after retrieval.**

11. **Stable experiments require freezing the corpus, benchmark queries, embedding model, and retrieval algorithm while changing one representation variable at a time.**

12. **The goal is not the largest possible context window. The goal is the smallest context that preserves the meaning required for retrieval.**

---

## Final Status

The first retrieval experiment series is complete:

```text
Whole Document
      ↓
Blind Chunking
      ↓
Structure-Aware V1
      ↓
Structure-Aware V2
```

The experiments successfully identified the next representation problem:

```text
V1:
dependency explosion

V2:
reduced dependency explosion
but possible definition explosion

Next:
bounded defined-term-chain enrichment
```

From there, improvements should increasingly move beyond chunk construction and into:

- hybrid retrieval,
- reranking,
- graph-assisted query-time context assembly,
- source-role / authority handling,
- retrieval evaluation at both document and clause level.
