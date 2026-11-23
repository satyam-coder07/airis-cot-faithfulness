# AIRIS Research Hackathon — CoT Faithfulness Reproduction & Extension

**Paper #2:** *Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting* (Turpin et al., NeurIPS 2023).

**Team:** Team Vengeance — Satyam Swarnakar, Japinder Kaur  
**Date:** November 23, 2026

## Overview

This repository reproduces and extends Turpin et al. (NeurIPS 2023) on 2026-era open-weight thinking models (`qwen3-30b-a3b` and `qwen3-32b`). By separating private `reasoning_content` from visible `content`, we audit whether chain-of-thought (CoT) unfaithfulness stems from internal bias adoption, selective public omission, or deliberation failure.

## Key Findings

- **Authority Framing Triggers Total Accuracy Collapse:** While modern thinking models resist casual user hints far better than 2023 baselines (accuracy drop of only -13.6% on 30B, p=0.25; -4.8% on 32B, p=1.00), framing the identical wrong answer as an "official expert grading rubric" collapses accuracy to 0.0% on valid pairs (-69.2% on 30B, p=0.0039; -66.7% on 32B, p=0.0312) with a 100% Bias-Following Rate (BFR).
- **Private-vs-Public Faithfulness Gap:** On authority bias-flips, models acknowledge the rubric in 88.9% (30B) and 100.0% (32B) of their private reasoning traces, but conceal it in 44.4% (30B) and 50.0% (32B) of public answers, fabricating independent justifications.
- **Turn-2 Self-Audit Falsification:** We pre-registered a prediction that models would deny hint influence in >60% of cases. The prediction was falsified: models admitted authority influence in 88.9% (30B) and 83.3% (32B) of cases upon direct follow-up, though 30B still denied casual hint influence 66.7% of the time.
- **Deliberation Paralysis:** Under authority conflict, parse failures more than doubled (8 -> 17 on 30B; 7 -> 16 on 32B) as models generated 6k-13k+ characters in private reasoning before exhausting the 3,072-token generation cap.

## Quickstart & Reproduction

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key
export HACKATHON_API_KEY="your_api_key_here"

# 3. Run full experiment across open models (10 items x 3 tasks = 30 items)
python3 run_experiment.py --models qwen3-30b-a3b qwen3-32b --n-per-task 10

# 4. Generate summary tables and figures
python3 analyze.py
```

## Repository Structure

```
├── run_experiment.py          # Main experiment harness (open models, 3 conditions, Turn-2 audit)
├── analyze.py                 # McNemar tests, bootstrap CIs, strict hint regex, figure generator
├── summary_metrics.csv        # Aggregate accuracy, BFR, hint acknowledgment, and denial rates
├── task_metrics.csv           # Task-level accuracy breakdowns (disambiguation_qa, movie_rec, snarks)
├── failure_audit.csv          # Automated failure pool (82 cases: flips, parse fails, other wrong)
├── failure_audit_35.csv       # 35 hand-audited failures across 6 behavioral categories
├── figures/
│   ├── accuracy_comparison.png # Figure 1: Unbiased vs. Biased CoT accuracy
│   └── faithfulness_audit.png  # Figure 2: BFR vs. internal/external hint acknowledgment
├── logs/
│   ├── smoke.jsonl            # Initial smoke run (max_tokens=1536 truncation test)
│   ├── fix_check.jsonl        # Verification run with max_tokens=3072
│   └── experiment_runs.jsonl  # Full experiment run (199 logged calls: 194 ok, 5 timeouts)
├── AIRIS_Report_Team_Vengeance.pdf  # 4-page academic research report
├── AIRIS_Slides_Team_Vengeance.pdf  # 7-slide presentation
├── requirements.txt           # Python package requirements
└── README.md                  # Project overview and reproduction instructions
```

## Dataset & Models

- **Dataset:** BIG-Bench Hard ([suzgunmirac/BIG-Bench-Hard](https://github.com/suzgunmirac/BIG-Bench-Hard)), tasks: `disambiguation_qa`, `movie_recommendation`, `snarks` (sampled with seed 42).
- **Models:** Open-weight models only: `qwen3-30b-a3b` (MoE) and `qwen3-32b` (Dense) served via local vLLM endpoint.

## AI Disclosure

Experiments were run strictly on local open models. Gemini, Claude Opus, and Antigravity IDE were utilized for experiment scaffold coding, analysis scripting, statistical cross-verification, and formatting. All prompt designs, failure taxonomies, and qualitative evaluations were performed and verified by the authors.

## Citation

```bibtex
@inproceedings{turpin2023language,
  title={Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting},
  author={Turpin, Miles and Michael, Julian and Perez, Ethan and Bowman, Samuel R},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={36},
  pages={74952--74965},
  year={2023}
}
```
