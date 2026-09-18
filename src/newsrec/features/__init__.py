"""A2 behavioural features (Q1). An explicit allowlist, never a denylist (D34).

Every feature here is computable from facts true strictly before the impression
it describes. Anything that is not - read_time, scroll_percentage, whole-catalogue
or whole-dataset aggregates - lives in `unavailable.py` as the Q9 ablation arm
and is never imported by the production feature set.
"""
