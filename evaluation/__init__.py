"""
Evaluation harness for the CSO project.

Three layers:

  Layer 1 (Day 8)  per-agent quality
                    - tool_use_accuracy:     auth agent uses the right tools

  Layer 2 (Day 9)  system quality
                    - decision_quality:      vs deterministic baseline
                    - latency_profile:       P50/P95/P99 by node

  Layer 3 (Day 9)  stress tests
                    - guardrail_stress:      adversarial input handling
                    - edge_case_pipeline:    20+ unusual transactions

Outputs land in evaluation/results/ as JSON + markdown report.
LangSmith integration is opt-in: set LANGCHAIN_API_KEY and the runner
will additionally upload datasets and post experiment results to your
LangSmith project. Without the key, everything still runs locally.
"""
