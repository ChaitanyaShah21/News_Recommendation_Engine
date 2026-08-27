"""Apply upstream develop's two fixes to a RELEASED compute_worker.py.

Mounting develop's whole file does not work: it requires Python 3.10+ (it calls a
staticmethod object directly, legal only from 3.10), and the released images run
Python 3.9. So we port the two fixes instead.

Written defensively because it edits code we cannot test until a real job arrives,
and a silently mis-patched worker destroys someone's submission:
  - anchors are regexes, not exact strings, so indentation or quoting differences
    do not cause a silent no-op;
  - every anchor is asserted, so a changed image layout fails the BUILD;
  - it is idempotent, so re-running cannot double-insert;
  - fix 2 derives the directory from metadata_path itself rather than assuming the
    attribute is called self.output_dir, which is the kind of guess that would
    still import cleanly and then fail hours later.
"""
import re
import sys

PATH = "/compute_worker.py"
src = open(PATH).read()
changed = []

# Fix 1 -- the secret arrives as a UUID (or a serialized UUID dict on older tags)
# and must be a string before it is logged or PATCHed back.
if "secret=str(run_args" in src:
    print("fix 1: already present")
else:
    m = re.search(r"^def run_wrapper\(run_args\):[ \t]*\n", src, re.M)
    assert m, "FATAL: run_wrapper(run_args) not found - image layout changed"
    src = src[:m.end()] + '    run_args.update(secret=str(run_args["secret"]))\n' + src[m.end():]
    changed.append("secret->str")

# Fix 2 -- result-only competitions run no program, so output/ never gets created,
# and push_output opens output/metadata unconditionally.
if "os.makedirs(os.path.dirname(metadata_path)" in src or "makedirs(self.output_dir" in src:
    print("fix 2: already present")
else:
    m = re.search(r"^([ \t]*)metadata_path\s*=\s*os\.path\.join\(.*\)[ \t]*\n", src, re.M)
    assert m, "FATAL: metadata_path assignment not found - image layout changed"
    indent = m.group(1)
    src = (src[:m.end()]
           + f"{indent}os.makedirs(os.path.dirname(metadata_path), exist_ok=True)\n"
           + src[m.end():])
    changed.append("mkdir output/")

if not changed:
    print("nothing to do")
    sys.exit(0)

open(PATH, "w").write(src)

# Prove it parses. A patch that produces a syntactically broken worker would
# otherwise only surface when a job is claimed - i.e. at someone else's expense.
import ast
ast.parse(open(PATH).read())
print("patched:", ", ".join(changed), "- and the file still parses")
