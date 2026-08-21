# Glossary

Every term defined once — plain language first, then technically.
Grows as we go. If a term is used anywhere in this project and is not here, that is a bug.

---

## Core vocabulary of the task

### Impression
**Plain:** One moment when a website showed somebody a list of headlines. Like one glance
at a newsstand — these particular papers were on display, at this particular time.

**Technical:** A single logged event containing a user identifier, a timestamp, the list
of candidate articles displayed (`article_ids_inview` in EB-NeRD, the `impressions` field
in MIND), and — in training data only — which of them were clicked.

---

### Candidate
**Plain:** One of the headlines that was on display in that glance.

**Technical:** An article present in an impression's display list. The set we must rank.
Typically 5–100 articles in EB-NeRD (average ~11), variable in MIND.

---

### Click label
**Plain:** Whether the person actually picked up that paper.

**Technical:** A binary target, 1 if clicked and 0 otherwise. In MIND it is encoded as a
suffix on the article identifier (`N55689-1` clicked, `N35729-0` not). In EB-NeRD it is a
separate list column `article_ids_clicked`. **Absent from both test sets** — that is what
we predict.

---

### History
**Plain:** The list of articles this person has read before now.

**Technical:** The user's prior click sequence. MIND stores it inline in every behaviour
row as a space-separated string. EB-NeRD stores it in a separate `history.parquet` file,
one row per user, as parallel lists (article identifiers, timestamps, read times, scroll
percentages).

---

### Retrieval vs re-ranking
**Plain:** Retrieval is walking into a library of 120,000 books and pulling the 100 that
might interest you. Re-ranking is being handed 11 books and putting them in the best order.

**Technical:** *Retrieval* (also *candidate generation*) searches the entire article corpus
and returns top-K, measured by recall@K. *Re-ranking* scores a supplied candidate list and
orders it, measured by ranking metrics. **This assignment needs both** — questions Q2/Q3
grade retrieval, the leaderboards grade re-ranking.

---

## Terms added per phase

_Phase 1 terms appear here once taught._
