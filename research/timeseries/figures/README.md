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

## Interactive presentation and fidelity

The page now contains two native SVG schematics, a native SVG break map, three native SVG histograms, and a vector horizon explorer. Every original graphic remains expandable and is the default without JavaScript. The schematics simplify residual connections and parallel heads, with explanatory block selection and the complete original diagrams available alongside them.

Both the pinned revision and repository HEAD were checked: they are the same revision. Neither notebook (`evaluations.ipynb` or `playground.ipynb`) exports the forecast tensor, break matrices, segment records, or trust-score array. Notebook chart outputs are PNGs. Consequently these interactive figures do **not** claim access to the original numerical results, and the forecasting experiment was **not rerun**.

`../scripts/read_figures.py` reads the local PNGs (Pillow and NumPy) and generates `../charts-data.js`. Run `python3 research/timeseries/scripts/read_figures.py` from the website root. Source-image hashes are embedded in the output. The conversion has these limits:

- Break positions: recover the colored cell in each of 24 rows of each heatmap. The matrix contains 604 entries; recovered positions lie on the source algorithm's five-entry grid. The first predicted position, 445, also appears numerically in the notebook. These positions remain identified as figure readings.
- Segment lengths: reconstruct `[break, 604-break]` for each horizon, then apply separate NumPy 20-bin histograms as in the source. Both series have 48 lengths. Verify all 40 frequencies independently against bar heights in the archived histogram.
- Segment slopes: recover 20 separate-range bins per series from the plotted color masks and frequency scale. All 40 integer frequencies are resolvable and sum to 48 per series. The horizontal bin boundaries remain approximate; no per-segment slopes are invented.
- Composite diagnostic: recover 30 original bin locations and heights from the saved histogram. Both boundaries and counts are approximate. At 270 pixels per 1,000 counts, small bars are below image resolution; the displayed ±4-count reading range describes image-reading precision, **not** a statistical interval. The reconstruction is not normalized to force a total of 6,034 and must not be presented as exact counts.
- Horizon explorer: the visible red and blue trace coverages are separated from the original source-over rendering on white, thresholded at 8% coverage, and converted to vector paths. Rectangle merging preserves the masks' geometry exactly; it does not interpolate new samples. The original legend is excluded, and isolated dashed break-guide pixels outside neighboring trace envelopes are removed before drawing independent vector guides. Horizontal and vertical scales are calibrated against the printed axes. Each source-image column spans roughly 21 entries, so hover shows a **visible trace range**, not an individual sample value. Antialiasing, occlusion, and image resolution prevent exact numeric recovery. Controls include independent series toggles, break guides, 1–16× horizontal zoom, drag panning, Shift+scroll zoom, and a keyboard/touch cursor. The experiment is not rerun. The panel uses native SVG; it no longer crops and enlarges a raster image onto a canvas.

CSV exports and accessible tables preserve these qualifications. Recovered figure readings must not be substituted for experiment outputs in downstream analysis. Existing article facts and original conclusions are unchanged.

## Content verification

- `evaluations.ipynb`: saved MASE 0.849; forecast shape (6034, 100, 24); 24 predicted and 24 observed breaks; one of each per horizon. Results were read, not rerun.
- `eval_pipeline.py`: pretrained traffic checkpoint; median for MASE; historical seasonal-naive scaling.
- `stats.py`: sample mean for segmentation; downsampling across forecast entries by 10 (604 retained entries); candidates 1–5; score is raw L2 cost + K log(n), not a fully specified likelihood-based BIC.
- `visualisations.py`: segment lengths and slopes are along entry order, not within a single sensor chronology. Original plot labels are retained and clarified in article captions.
- Unlike the README’s interpretation, saved results do not establish over-segmentation (equal counts; minimum candidate selected), and the slope figure shows more extreme observed tails.
- Trust score tests forecast distributions and mean levels, not residuals. It is an average of two test fractions and a binary indicator, not a calibrated probability. Errors return zero.
- Article uses the shared SAFE layout at `research/safe/project.css`; article-specific styles and interactions live in this directory.
