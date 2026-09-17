# SAFE figure provenance

All six figures are from the paper and its public code repository. Each repository PDF was visually matched to the corresponding figure in [arXiv v1](https://arxiv.org/pdf/2609.17865v1).

Source repository commit: `a099eb8323e5d826cf6b8b4e4cefbfdab0e69d9e`. No repository clone was created. Only the selected figure PDFs were downloaded for inspection and rendering.

Web assets are lossless WebP renders of the original PDFs, produced with Poppler at a 2400-pixel longest edge. There is no cropping, redrawing, recoloring, or alteration of labels or data. Figures retain their paper numbering; Figure 1 and `imgs/Safe.png` are not used on this page.

| Web asset | Paper figure / PDF page | Original PDF | Original PDF SHA-256 |
| --- | --- | --- | --- |
| `severity-probability.webp` | Figure 2 / page 5 | [main_experiment/graphs/post/decision_rates_by_severity.pdf](https://github.com/omertafveez-2001/Do-Models-Seek-Safety-Evidence/blob/a099eb8323e5d826cf6b8b4e4cefbfdab0e69d9e/main_experiment/graphs/post/decision_rates_by_severity.pdf) | `8ad702b9de99a3e7666ae429bdaa180144677d74d16ff01de5dbbd738d378cb1` |
| `inspection-cost.webp` | Figure 3 / page 5 | [main_experiment/graphs/inspect/rate_curves_by_cost.pdf](https://github.com/omertafveez-2001/Do-Models-Seek-Safety-Evidence/blob/a099eb8323e5d826cf6b8b4e4cefbfdab0e69d9e/main_experiment/graphs/inspect/rate_curves_by_cost.pdf) | `b4c6cac04c56607ce3ac4cad070a2f3b867f645fbd01bb410ec28b8def95f71f` |
| `post-inspection.webp` | Figure 4 / page 5 | [main_experiment/graphs/post/post_decision_distribution_percentage.pdf](https://github.com/omertafveez-2001/Do-Models-Seek-Safety-Evidence/blob/a099eb8323e5d826cf6b8b4e4cefbfdab0e69d9e/main_experiment/graphs/post/post_decision_distribution_percentage.pdf) | `064e92411e9b8d4e599747db55275be600488d7f1f230873dcfa97c1ca15f8c7` |
| `cost-obligation.webp` | Figure 6 / page 7 | [ablation/cost_obligation_decomposition/graphs/inspection_rate_cost_decomposition.pdf](https://github.com/omertafveez-2001/Do-Models-Seek-Safety-Evidence/blob/a099eb8323e5d826cf6b8b4e4cefbfdab0e69d9e/ablation/cost_obligation_decomposition/graphs/inspection_rate_cost_decomposition.pdf) | `31c6009899f48e6edf16ff5446028cdef2a1a50ab77b5d3816d5e9b7d0e56f70` |
| `counterfactual-decisions.webp` | Figure 7 / page 7 | [ablation/cot_faithfullness/graphs/flip_rate_by_edit_type.pdf](https://github.com/omertafveez-2001/Do-Models-Seek-Safety-Evidence/blob/a099eb8323e5d826cf6b8b4e4cefbfdab0e69d9e/ablation/cot_faithfullness/graphs/flip_rate_by_edit_type.pdf) | `2c776d233ae2e5b26c66d952b0f4bf0cc63e43109aa0c8f3b1b9b15ca23c3fde` |
| `rationale-acknowledgement.webp` | Figure 8 / page 8 | [ablation/cot_faithfullness/graphs/acknowledgement_rate_by_edit_type.pdf](https://github.com/omertafveez-2001/Do-Models-Seek-Safety-Evidence/blob/a099eb8323e5d826cf6b8b4e4cefbfdab0e69d9e/ablation/cot_faithfullness/graphs/acknowledgement_rate_by_edit_type.pdf) | `a837ab456eb5bacfff53f41eccc850c050eb06beb6711624704b7d93c58dcb96` |

Content references: paper §§3–4 for benchmark design and main results, §5 for the 50-scenario decomposition, §6 and Appendix A.7 for counterfactual diagnostics, §7.1 and Appendices A.4–A.6 for limitations. Avoid substituting the counterfactual flip rates for population effects or equating conditional halt rates with overall safety.
