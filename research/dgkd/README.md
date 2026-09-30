# Decoupled Gradient Knowledge Distillation article

Restyled from the site's original `blogs/decoupled_gradient_kd/dkd.md` (October 26, 2025). The original text remains available in Git history. The article and assets now live exclusively under `research/dgkd/`; the old blog page has been removed.

- Shared visual foundation: `../safe/project.css` and `../../style.css`.
- Shared figure and section-navigation behavior: `../timeseries/project.js`.
- Article-specific styles: `project.css`.
- Equations use HTML text, subscripts, and superscripts and require no external math renderer.
- All original blog image assets have moved into `figures/`. The article uses `training-comparison.jpeg` (renamed from `cifar100_training_comparison.jpeg`); `gradient.png` and `logits_graphs.jpeg` are retained as project assets.
- `figures/training-comparison.jpeg` is unchanged. Its labels describe ResNet56 → ResNet20, ε = 1.0, over 100 epochs; it is not the ResNet50 → ResNet18 gradient-norm experiment described in the text.
- The existing motivation, hypotheses, reported observations, training caveats, limitations, and future-work sections are retained. The sign explanation is corrected: a positive squared-distance term in a minimized objective penalizes differences. Unsupported claims of guaranteed non-vanishing MSE gradients, direction neutrality, and a proved sustaining mechanism are qualified.
- No training experiments were rerun as part of this presentation update.
