---
layout: post
title: "Bringing Balance Back into Logit Distillation: Decoupled Gradient Knowledge Distillation"
date: 2025-10-26
permalink: /blogs/dkd/
---

In recent times, knowledge distillation has become a go-to technique for transferring the “wisdom” of a large, well-trained teacher model into a smaller, more efficient student model. The question we ask: how best to transfer not just “what” the teacher predicts but also “how” it reasons?  

In this blog post we present our work on **Decoupled Gradient Knowledge Distillation (DGKD)** — an implementation inspired by the paper [Decoupled Knowledge Distillation (Zhao et al., 2022)](https://arxiv.org/abs/2203.08679) — and share both the intuition, our modifications, and takeaways.

### 🧠 Background: Why "Decoupled" Knowledge Distillation

The original DKD paper shows that the standard KD loss (e.g., the KL divergence between teacher and student logits) can be broken down into two parts:

- **TCKD (Target-Class Knowledge Distillation):** focuses on the predicted target class — how confident the teacher is on the “correct” class.  
- **NCKD (Non-Target-Class Knowledge Distillation):** covers all other classes, i.e., the “dark knowledge” of how the teacher distributes probability mass among wrong classes.

The key insights from the paper are:
