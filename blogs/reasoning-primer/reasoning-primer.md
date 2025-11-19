---
layout: post
title: "Reasoning: Primer on Reasoning Prompts in Large Language Models"
date: 2025-10-26
permalink: /blogs/reasoning-primer/
---

<p style="text-align: center;">
  "Large Language Models are pattern matchers and stochastic machines."
</p>
This is what I thought initiailly, when I started to work on Language Models. But there is much to it. In the past 5 years, we have seen plenty of work in AI Safety, and Reasoning. So one could really ask, <i>"What exactly is Reasoning"?</i> or <i>"What is AI Safety in LLMs"?</i> <br> <br>

> "**Reasoning** in LLMs refers to the model's ability to perform structured, step-wise, logically coherent inference that goes beyond surface-level pattern matching, allowing the model to solve problems via multi-step computation, planning, abstraction, and generalization" ~ <i>GPT-5 Instant Response</i> <br> <br>

Well there is a reason why, I deliberately attach a GPT response, because maybe MAYBE... GPT-5 does not <i>reason<i> that well. And that is because this definition sounds *true* but is definitely not academically *true*. Let us also think step-by-step. <br> <br>

1. How does Human Reasoning differs compared to LLM Reasoning <br>
Humans use neural-symbolic reasoning: a mixture of intuition and explicit symbolic manipulation. The brain supports working memory, goal-directed planning, and causal models. We can apply rules, logic, and abstraction deliberately. <br>
LLMs perform next-token prediction using a statistical function learned from large corpora. All "reasoning" emerges from pattern completition, not explicit symbolic rules. LLMs do not maintain persistent working memory; they rely on the context window and hidden activations. 

2. Reasoning in LLMs is not as simple.<br>
*Setting*: Imagine you are sitting in an exam, and for some very *weird* reason, you somehow knew that the question you are about to attempt is almost the same as the question from previous examination (maybe in the previous year or semester), so you try to copy the previous answer that you had memorized to ensure it resembles marking scheme. <br>
In a very similar fashion, in LLMs, we call it **Reward Hacking**. That is, that LLMs somehow learns to bypass the cost function by hacking the problem, and pretending that it is solving the problem. Sounds a bit *confusing* right? It was confusing to me as well. <br>
Let's take a coding problem: 

``` python
def add(a, b, target):
  return (a + b) == target
```

Suppose this is the function that the LLM is supposed to generate. It may fail some tests, and it might not. But what if it generates this
```python
def add(a, b, target):
  return True
```
It probably will pass more than 50% of the test cases. This is reward hacking. 



