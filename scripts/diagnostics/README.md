# Diagnostics

Measurement scripts that are **not** pipeline steps: they produce the numbers quoted in
`reports/NUMBERS.md` section K and the reasoning in `ARCHITECTURE.md` D42–D44.

They live here, in the repository, because the first versions lived in `/tmp` and were
destroyed when WSL restarted on 2026-09-20 — taking with them the only record of *how*
several published figures were produced. The ledger promises every quantity traces to a
command; a command that no longer exists does not satisfy that.

| script | produces | quoted in |
|---|---|---|
| `rack_variance.py` | between-rack variance share per feature; rack size profile | NUMBERS K.1, D42 |
| `rack_arms.py` | the 4-arm validation sweep that selected `+both` | NUMBERS K.2, D42a |
| `verify_rebuild.py` | proof the D42 rebuild changed no base column | NUMBERS K.3 |
| `time_word_nrms.py` | word-level vs sentence-level training cost | NUMBERS K.4, D43a |
| `profile_build_rows.py` | where the batch feature path spends its time | NUMBERS K.5, D44/D44a |
| `mutate.py` | mutation testing harness (clears `__pycache__`, per the 2026-08-25 error-log entry) | D42's "13 of 13 caught" |

All are read-only with respect to the feature store and write nothing to `reports/`.
