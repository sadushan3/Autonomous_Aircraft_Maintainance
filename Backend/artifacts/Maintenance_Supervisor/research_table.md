# Autonomous Maintenance Supervisor - Research Benchmark Tables

## Table 1: Sequential RL Policy Optimization & Decision Performance (Proposal Section 3.6 Metrics)

| Decision Policy | Failure Prevention Rate | Catastrophic Failures | Mean Maintenance Cost ($) | Mean Downtime (Hours) | Cost Reduction vs Reactive | Downtime Reduction vs Reactive |
|---|---|---|---|---|---|---|
| **PPO Champion Policy (Proposed)** | **100.0%** | **0** | **$500.00** | **12.0h** | **78.98%** | **68.89%** |
| Rule-Cascade Baseline | 100.0% | 0 | $3,159.29 | 12.0h | — | — |
| Reactive (TBO) Baseline | 67.9% | 9 | $2,378.57 | 38.6h | Base | Base |

## Table 2: Multi-Model Supervised & Baseline Comparison

| Model Architecture | Accuracy | Macro F1 | False Critical Rate | Missed Critical Rate | Economic Cost Penalty |
|---|---|---|---|---|---|
| Majority Vote Baseline | 0.5959 | 0.1494 | 0.0000 | 1.0000 | $396,096.00 |
| Rule-Based Baseline | 1.0000 | 1.0000 | 0.0000 | 0.0000 | $0.00 |
| RUL-Only Baseline | 0.9985 | 0.9959 | 0.0000 | 0.0223 | $305.00 |
| Risk-Only Baseline | 0.6971 | 0.4785 | 0.0000 | 0.3971 | $58,550.00 |
| Anomaly-Only Baseline | 0.6100 | 0.2430 | 0.0000 | 0.7439 | $172,796.00 |
| Champion Supervisor Agent | 0.7445 | 0.5378 | 0.0000 | 0.0000 | $49,835.00 |
