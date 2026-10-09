"""Regenerate excluded_rows.json from the numerical split's parquet.

Excludes numerical rows whose reference states that no valid value exists
("No feasible integer N satisfies all three constraints.", "none",
"Target cannot be achieved ..."). math-verify reads such a sentence as a
product of single-letter symbols, so the numerical grader credits only a reply
that repeats the reference's wording, and credits a reply equal to any
problem parameter the sentence quotes ("the ratio never reaches 1.5").

Usage: python build_excluded_rows.py [path/to/principia_collection_numerical.parquet]
"""

import json
import re
import sys

import pyarrow.parquet as pq

# \text{...}-style wrappers keep their contents; other LaTeX commands and
# delimiters are dropped before matching.
_TEXT_WRAPPER = re.compile(r"\\(?:text|mathrm|textbf|operatorname)\s*\{([^{}]*)\}")
_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+|[\\$(){}\[\]]")
# A reference that opens by denying that a value exists.
_LEADING_DENIAL = re.compile(
    r"^\W*(no|none|not|there\s+(is|are|exists?)\s+no|does\s+not|doesn't|cannot|can't|impossible"
    r"|infeasible|undefined|never|nonexistent|non-existent|n/a|no\s+such)\b",
    re.I,
)
# A denial later in the sentence ("Target cannot be achieved ...").
_DENIAL = re.compile(
    r"\b(no\s+(such|feasible|solution|real|positive|integer|finite|valid|value)|does\s+not\s+exist"
    r"|cannot\s+be|impossible|infeasible|not\s+(attainable|achievable|possible|feasible)"
    r"|never\s+(exceeds|reaches))\b",
    re.I,
)


def states_no_value(reference: str) -> bool:
    text = _LATEX_COMMAND.sub(" ", _TEXT_WRAPPER.sub(r" \1 ", reference))
    return bool(_LEADING_DENIAL.search(text) or _DENIAL.search(text))


def main(path: str) -> None:
    answers = pq.read_table(path, columns=["answer"]).column("answer").to_pylist()
    rows = [i for i, answer in enumerate(answers) if states_no_value(answer)]
    with open("excluded_rows.json", "w") as f:
        json.dump({"train_numerical": rows}, f, separators=(",", ":"))
        f.write("\n")
    print(f"excluded {len(rows)} of {len(answers)} numerical rows")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "principia_collection_numerical.parquet")
