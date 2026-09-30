# Article figures and source notes

Figures downloaded individually from revision `eb46a634c309312aff3721841beb81142d0dac4e` (June 16, 2025). No repository clone was made. PNGs are unmodified.

| Local file | Source | SHA-256 |
| --- | --- | --- |
| `architecture.png` | [Original PNG](https://github.com/omertafveez-2001/Transformers-for-Timeseries-A-Statistical-View/blob/eb46a634c309312aff3721841beb81142d0dac4e/imgs/autoformer_architecture.png) | `17feb02529e42a596871be6b47e6bab0aa6ecca345210328236f1c35693c11d7` |
| `autocorrelation.png` | [Original PNG](https://github.com/omertafveez-2001/Transformers-for-Timeseries-A-Statistical-View/blob/eb46a634c309312aff3721841beb81142d0dac4e/imgs/autoformer_autocorrelation_only_attention.png) | `007dcfb347ab1c00cb938baf2aef87943dd087267da4a36692d08a2bb12010e4` |
| `break-heatmap.png` | [Original PNG](https://github.com/omertafveez-2001/Transformers-for-Timeseries-A-Statistical-View/blob/eb46a634c309312aff3721841beb81142d0dac4e/imgs/results/pred_breaks_hour.png) | `e678b3a82027fb4d860365fe198f68a00362e8d3c69e3100a29e7fb1b5b6aa4f` |
| `horizon-overlays.png` | [Original PNG](https://github.com/omertafveez-2001/Transformers-for-Timeseries-A-Statistical-View/blob/eb46a634c309312aff3721841beb81142d0dac4e/imgs/results/breaks_per_hour.png) | `955738ab4eb0e2d5594484eb854051fef4902fbdeb7fb3e1528c3f57a5663a12` |
| `segment-lengths.png` | [Original PNG](https://github.com/omertafveez-2001/Transformers-for-Timeseries-A-Statistical-View/blob/eb46a634c309312aff3721841beb81142d0dac4e/imgs/results/segment_length_distr.png) | `5b2cf15d59128d22989c9a121d536957d793d77c57fd43a264c7a852d6c02f32` |
| `segment-trends.png` | [Original PNG](https://github.com/omertafveez-2001/Transformers-for-Timeseries-A-Statistical-View/blob/eb46a634c309312aff3721841beb81142d0dac4e/imgs/results/segment_trend_dist.png) | `a2778c42948b60f1ed12323a08eee9de41de616d600ed0a088c4c2649313368f` |
| `trust-score.png` | [Original PNG](https://github.com/omertafveez-2001/Transformers-for-Timeseries-A-Statistical-View/blob/eb46a634c309312aff3721841beb81142d0dac4e/imgs/results/trust_score.png) | `f37ec1cad5aadac8447f14f79ff3003b77587cd459a6a0ce950f1bc638cc28de` |

Architecture diagrams describe Wu et al., [Autoformer (2021)](https://arxiv.org/abs/2106.13008), reproduced by the project README. Result plots are from the project repository.

## Content verification

- `evaluations.ipynb`: saved MASE 0.849; forecast shape (6034, 100, 24); 24 predicted and 24 observed breaks; one of each per horizon. Results were read, not rerun.
- `eval_pipeline.py`: pretrained traffic checkpoint; median for MASE; historical seasonal-naive scaling.
- `stats.py`: sample mean for segmentation; downsampling across forecast entries by 10 (604 retained entries); candidates 1–5; score is raw L2 cost + K log(n), not a fully specified likelihood-based BIC.
- `visualisations.py`: segment lengths and slopes are along entry order, not within a single sensor chronology. Original plot labels are retained and clarified in article captions.
- Unlike the README’s interpretation, saved results do not establish over-segmentation (equal counts; minimum candidate selected), and the slope figure shows more extreme observed tails.
- Trust score tests forecast distributions and mean levels, not residuals. It is an average of two test fractions and a binary indicator, not a calibrated probability. Errors return zero.
- Article uses the shared SAFE layout at `research/safe/project.css`; article-specific styles and interactions live in this directory.
