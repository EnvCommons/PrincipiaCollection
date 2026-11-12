# PrincipiaCollection

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/PrincipiaCollection) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/facebook/principia-collection)

## Description

**PrincipiaCollection** is a large-scale training environment for STEM mathematical derivation via RL. It contains about 552K synthetic problems across two grading modes: mathematical objects (LLM-judged) and numerical answers (exact match).

## Capabilities

- Mathematical derivation and symbolic reasoning
- Numerical computation
- STEM knowledge across diverse mathematical topics

## Compute Requirements

No sandbox is used. The environment grades a single text reply.

## License

[MIT](https://opensource.org/licenses/MIT).

## Tasks

There are 248,743 tasks in the `train` split and 302,962 in `train_numerical`.

- **train**: 248,743 mathematical object problems (LLM-judged)
- **train_numerical**: 302,962 numerical problems (math-verify grading). The source has 305,656; the 2,694 whose reference states that no valid value exists (e.g. "No feasible integer N satisfies all three constraints.") are left out, since numerical grading can't credit a correct reply to them. `excluded_rows.json` lists them by source row; `build_excluded_rows.py` regenerates it.
- Each task has: `id`, `problem_statement`, `topic`, `answer_type`, `split`
- Answer types include: Set, Interval, Equation, Inequality, Matrix, Integer, Decimal, Fraction

## Reward Structure

Binary reward (0.0 or 1.0). A repeated submission in the same episode is not re-graded and gets -0.1.
- `train` split: a single LLM equivalence judge call to `gpt-5-mini`
- `train_numerical` split: math-verify equivalence. A decimal reference with at least 2 significant figures (e.g. `0.0011`, `57.0 mK`, `9.1e-6`) also accepts any answer that rounds to it at those significant figures (`0.00112`); integer and fraction references stay exact

## Data

Source: [facebook/principia-collection](https://huggingface.co/datasets/facebook/principia-collection) on HuggingFace. Two parquet files (mathematical_object and numerical splits), hosted on the OpenReward platform.

## Tools

None. The model is given no tools: it answers the problem as an ordinary message, and that message ends the rollout.

Grading runs through a hidden terminal tool (`submit`): numerical answers are checked with math-verify (LaTeX-aware, tolerant of \\boxed{} and other common formats), and non-numerical answers are judged by an LLM for equivalence.

## Time Horizon

Single-turn. The episode ends with the model's first reply.

## Other Environment Requirements

- OpenAI API key required for `train` split (passed via `secrets["openai_api_key"]`)
- No API key needed for `train_numerical` split

## Safety

No safety concerns: the environment grades mathematical derivations only.

## Citations

```bibtex
@misc{aggarwal2026reasoningmathematicalobjects,
      title={Reasoning over mathematical objects: on-policy reward modeling and test time aggregation},
      author={Pranjal Aggarwal and Marjan Ghazvininejad and Seungone Kim and Ilia Kulikov and Jack Lanchantin and Xian Li and Tianjian Li and Bo Liu and Graham Neubig and Anaelia Ovalle and Swarnadeep Saha and Sainbayar Sukhbaatar and Sean Welleck and Jason Weston and Chenxi Whitehouse and Adina Williams and Jing Xu and Ping Yu and Weizhe Yuan and Jingyu Zhang and Wenting Zhao},
      year={2026},
      eprint={2603.18886},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
}
```
