# PrincipiaCollection

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/PrincipiaCollection)

## Description

**PrincipiaCollection** is a large-scale training environment for STEM mathematical derivation via RL. It contains 554K synthetic problems across two grading modes: mathematical objects (LLM-judged) and numerical answers (exact match).

## Capabilities

- Mathematical derivation and symbolic reasoning
- Numerical computation
- STEM knowledge across diverse mathematical topics

## Compute Requirements

- Mathematical object split: requires OpenAI API access for LLM-based equivalence judging
- Numerical split: no external API needed (exact match grading)

## Tasks

- **train**: 248,743 mathematical object problems (LLM-judged)
- **train_numerical**: 305,656 numerical problems (exact match)
- Each task has: `id`, `problem_statement`, `topic`, `answer_type`, `split`
- Answer types include: Set, Interval, Equation, Inequality, Matrix, Integer, Decimal, Fraction

## Reward Structure

Binary reward (0.0 or 1.0).
- `train` split: single LLM equivalence judge call
- `train_numerical` split: exact numeric match with small tolerance

## Data

Source: [facebook/principia-collection](https://huggingface.co/datasets/facebook/principia-collection) on HuggingFace. Two parquet files (mathematical_object and numerical splits). Mounted at `/orwd_data` in production.

## Tools

None. The model is given no tools: it answers the problem as an ordinary
message, and that message ends the rollout.

Grading runs through a hidden `@terminal` tool: numerical answers are checked
with math-verify (LaTeX-aware, tolerant of \\boxed{} and other common
formats), and non-numerical answers are judged by an LLM for equivalence.

## Time Horizon

Single-turn. One tool call per episode.

## Environment Difficulty

Ranges from introductory to advanced undergraduate across diverse mathematical topics.

## Other Environment Requirements

- OpenAI API key required for `train` split (passed via `secrets["openai_api_key"]`)
- No API key needed for `train_numerical` split

## Safety

No safety concerns — environment grades mathematical derivations only.

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
