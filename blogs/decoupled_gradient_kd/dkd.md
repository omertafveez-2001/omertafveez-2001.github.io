---
layout: post
title: "Bringing Balance Back into Logit Distillation: Decoupled Gradient Knowledge Distillation"
date: 2025-10-26
permalink: /blogs/decoupled_gradient_kd/
---
![gradients](./imgs/gradient.png)
In recent times, knowledge distillation has become a go-to technique for transferring the “wisdom” of a large, well-trained teacher model into a smaller, more efficient student model. The question we ask: how best to transfer not just “what” the teacher predicts but also “how” it reasons?  

In this blog post we present our work on **Decoupled Gradient Knowledge Distillation (DGKD)** — an implementation inspired by the paper [Decoupled Knowledge Distillation (Zhao et al., 2022)](https://arxiv.org/abs/2203.08679) — and share both the intuition, our modifications, and takeaways.

## 🧠 Background: Why "Decoupled" Knowledge Distillation

The original DKD paper shows that the standard KD loss (e.g., the KL divergence between teacher and student logits) can be broken down into two parts:

- **TCKD (Target-Class Knowledge Distillation):** focuses on the predicted target class — how confident the teacher is on the “correct” class.  
- **NCKD (Non-Target-Class Knowledge Distillation):** covers all other classes, i.e., the “dark knowledge” of how the teacher distributes probability mass among wrong classes.

The key insights from the paper are:
- The "dark knowledge" (NCKD) is very important for good student performance; yet in many classical KD formulations it gets supressed. 
- By decoupling the two components, we gain more flexibility: we can can tune how much we emphasis TCKD vs NCKD. 

In short: we do not just want the student to "copy" the teacher's most-likely class, but also to learn the structure of confusion the teacher holds among the wrong answers. 

## 🚀 Our Project: Implementation + Extension
We built our codebase around the repository at github.com/omertafveez-2001/Decoupled-Gradient-Knowledge-Distillation
, implementing the core DKD loss and then experimenting with a novel coupling mechanism between the two branches of the loss.

### What we did differently
- Instead of letting TCKD and NCKD purely operate independently, we introduced a coupling term: the mean-squared error (MSE) between gradients of TCKD and NCKD with respect to the student logits (or student model parameters).

- Interestingly, when we maximize this gradient‐MSE term (i.e., encourage the gradients of TCKD and NCKD to diverge / be different), we observed better performance.

- The effect:
    - The student’s target‐class logit confidence saturates (i.e., once it's confidently correct, further boosting confidence doesn’t shrink the loss further).
    - The student is strongly penalised (via higher loss) when the student is correct but the teacher is not — i.e., the model is forced to stay loyal to the teacher’s logit structure, not just prediction accuracy.
    - We observed faster “neural collapse” (class centres become tight, intra‐class variance shrinks) and more compact representations.

### Original Loss Function
$$
\text{KD} = \text{KL}(b^{\tau} || b^{s}) + (1-p_{t}^{\tau})\text{KL}(\hat{p}^{\tau} || \hat{p}^{s})
$$
<br> 
where the first KL term is the similarity between the binary probabilities between teacher and student. This is called *Target Class Knowledge Distillation*. Meanwhile, the second KL term represents the similarity between the teacher's and student's probabilities among non-target class which is *Non-Target Class Knowledge Distillation*

### <span style="color:green;">Updated</span> Loss Function
$$
\text{KD} = \text{KL}(b^{\tau} || b^{s}) + (1 - p_{t}^{\tau})\, \text{KL}(\hat{p}^{\tau} || \hat{p}^{s}) + \alpha \, \frac{1}{n} \sum_{i=1}^{n} \left\| \nabla_{z_i^{s_{tc}}} \ell_t - \nabla_{z_i^{s_{ntc}}} \ell_s \right\|_2^2
$$

- $$\nabla_{z_i^{s}} \ell_t$$ is the student loss with respect to target class logits
- $$\nabla_{z_i^{s_{ntc}}} \ell_s$$ is the student loss with respect to non target class logits.

### Why might this work?
- By maximising the gradient‐MSE, we force TCKD and NCKD to provide distinct training signals rather than redundant ones. This drives the student to learn complementary features: one focusing on sharpening target‐class confidence, the other on modeling the non‐target distribution structure.
- The saturation of the target‐class confidence prevents the student from becoming over‐confident (which often harms generalization).
- The strong alignment with teacher logit structure helps the student internalize the teacher’s view of class‐relationships — not just the final correct label.

### Why MSE (and specifically rescaled MSE) is surprisingly good? 
Mean-Squared Error wasn’t my initial choice. I experimented with several entropy-based measures—including cosine similarity, KL-divergence, and Wasserstein distance—but none matched the stability and performance of MSE. My perspective shifted after reading [On the Optimization Landscape of Neural Collapse under MSE Loss: Global Optimality with Unconstrained Features](https://arxiv.org/pdf/2203.01238)
, which provided a rigorous theoretical foundation explaining why MSE can be remarkably effective for classification tasks.

#### About the paper
In their paper, they examine the behaviour of deep networks trained with the MSE loss on classification. Their key results provide theoretical and empirical motivations for using MSE and help explain why our approach is effective. 
- They show that when a network is trained under unconstrained-features model with MSE loss, the only global minimisers correspond to the **Neural Collapse**: within-class features collapse to the class-mean, cleas means form a simplex equi-angular tight frame (ETF) and classifiers align with these means. 
- They further analyse a rescaled version of MSE and show that rescaling improves the optimisation landscape: ie, it makes the saddle points sharper, the descent path more stable, and convergence to the NC solution faster. 
- Empirically, they observe that classification models trained with MSE can perform on-par or even better than CE/BCE.
- They note that MSE tends to produce faster neural collapse compared to CE. 

Here are the intuitive reasons why MSE or scaled MSE may offer advantages:
- <u>*Smooth gradient structure*</u>: MSE uses squared-error between the logit output and the target vector. This produces continuous, smooth gradients for all output dimensions. In contrast BCE/CS emphasises the targt class heavily, and rapdidly drives the student to push the target probability near one and the others zero. The gradient contributions from non-target classes diminish fast. <br>
As seen in our experiments, when we want non-target information to still matter, MSE's smoother gradient contributions help maintain signals from the non-target logits.
- <u>*Encourgement of feature collapse and class-structure*</u>: The NC theory shows that MSE pushes features to collapse in a very clean geometric structure (simplex ETF). Because the loss equally penalises all coordinates of the output, the network is encouraged to treat all classes (target + non-target) in a balanced way. This leads to tighter clusters and maximal separation. <br>
In our work we observed more compact intra-class clusters and faster neural collapse — this is unsurprising since we introduced a loss coupling (maximising gradient‐MSE) that encourages diversified gradients between the target‐ and non‐target branches. The foundation of that effect is arguably supported by the MSE landscape results.
- <u>*Rescaling improves landscape and generalisation*</u>: Zhou et al. find that a rescaled MSE — e.g., multiplying the loss by a constant or adjusting the target vector scaling — improves convergence behaviour. This is because the magnitude of gradients and the curvature of the loss surface can be better controlled, reducing plateaus and sharp minima. <br>
In our adaptation, by explicitly applying an MSE term between gradients (and effectively scaling it by a coupling coefficient ε) we are leveraging this notion of controlled magnitude and structured gradients. The fact our model performs well by maximising the gradient MSE aligns with the idea that certain directions in the gradient space should be emphasised (diverged) to aid representation learning.

### ✅ How this supports our methodological choice
Because we introduced a coupling term that maximises the MSE between gradients of TCKD and NCKD, we are effectively:

- Keeping gradients from the two branches active and diverse, which matches the smooth and balanced gradient regime that MSE enables.

- Leveraging the faster neural collapse and improved geometry that MSE offers, which you observed in more compact intra‐class representations.

- Avoiding the overconfidence / flat gradient issues that can accompany CE/BCE loss for distillation (where once the student is confident, non‐target gradient signals vanish).

- Employing a scaled-MSE style approach (via the coupling hyper-parameter ε) that emphasises structural gradient differences rather than just output matching, echoing the rescaling insight of Zhou et al.



## 📊 Performance Comparison: Gradient-Decoupled KD vs. Decoupled KD

| Dataset                                 | Gradient-Decoupled KD | Decoupled KD |
| --------------------------------------- | --------------------- | ------------ |
| CIFAR-100 — ResNet50 → ResNet18         | **62.01%**            | 61.07%       |
| CIFAR-100 — ViT-S → ResNet18            | **58.64%**            | 57.31%       |
| CIFAR-100 — Self-Distillation ResNet18  | **62.35%**            | 61.53%       |
| CIFAR-100 — Self-Distillation ResNet50  | 53.39%                | **54.31%**   |
| CIFAR-100 — Self-Distillation ResNet101 | **55.94%**            | 48.99%       |
| CIFAR-100 — ShuffleNetV2 → ResNet50     | 55.49%                | **56.48%**   |
| CIFAR-100 — ShuffleNetV2 → ViT-S        | 55.56%                | **55.84%**   |
| ImageNet — ResNet50 → ResNet18          | **48.42%**            | 45.48%       |
| ImageNet — ResNet50 → ShuffleNet        | 44.90%                | **45.59%**   |
| ImageNet — ViT-S → ResNet18             | **41.81%**            | 41.57%       |
| ImageNet — ViT-S → ShuffleNet           | **39.74%**            | 39.42%       |
| ImageNet — Self-Distillation ResNet101  | **30.49%**            | 0.50%        |
| ImageNet — Self-Distillation ResNet50   | **48.09%**            | 48.06%       |
| ImageNet — Self-Distillation ResNet18   | **48.29%**            | 46.40%       |

✅ Bold entries indicate the student outperforming the original DKD baseline <br>
📌 DGKD improves on 10 out of 14 evaluated distillation settings

From the table above, we infered that our variant was not successful in ourperforming Decoupled KD in mobile networks or smaller CNN models such as ShuffleNetV2 and MobileNet, however it was still consistent across 10/14 experiments.

### 🔍 Key Observations & Takeaways

From our experiments we noted:

- **Loss behaviour**: When the student’s target class logit is sufficiently confident (say probability 0.9 vs teacher 0.8) the overall loss doesn’t shrink further, i.e., it plateaus instead of chasing infinite confidence.

- **Teacher vs student mismatch penalty**: If the student predicts correctly but the teacher was less confident (or wrong) the coupling term spikes the loss. That means we’re enforcing fidelity to teacher’s distribution even when accuracy is achieved.

- **Representation compactness**: We saw that intra‐class features became more tightly clustered in embedding space compared to baseline KD methods, implying better generalization and faster convergence.

### Logits Comparison between Teacher & DKD vs Teacher & Decoupled Gradient KD
![Gradient Flow Diagram](./imgs/logits_graphs.jpeg)

We computed the Mean Squared Distance between teacher and the variants. The metrics are as follows:
- Teacher and DKD: 43.884
- Teacher and DGKD: 34.666

So our logits were significantly closer. 

## Future Work
- Disect the loss function and complete its derivation to further develop the understanding of the model's architectural differences (benefits and disadvantages). In doing so, it is highly important to understand the role of Non Target Class and Target Class Logits in this space.

- Continue the experimentation across more complex datasets in image classification: Animal10, and OOD Generalization (Scrambled, Noisy)

- Continue the experimentations across object detection datasets: MS COCO
