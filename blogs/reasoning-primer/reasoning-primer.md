---
layout: post
title: "Reasoning: Primer on Reasoning Prompts in Large Language Models"
date: 2025-10-26
permalink: /blogs/reasoning-primer/
---
![Reasoning-image](./imgs/reasoning_title.jpg)
<p style="text-align: center;">
  "Large Language Models are pattern matchers and stochastic machines."
</p>
This is what I thought initiailly, when I started to work on Language Models. But there is much to it. In the past 5 years, we have seen plenty of work in AI Safety, and Reasoning. So one could really ask, <i>"What exactly is Reasoning"?</i> or <i>"What is AI Safety in LLMs"?</i> <br> <br>

> **Reasoning** in LLMs refers to the model's ability to perform structured, step-wise, logically coherent inference that goes beyond surface-level pattern matching, allowing the model to solve problems via multi-step computation, planning, abstraction, and generalization ~ <i>GPT-5 Instant Response</i> <br> <br>

Well there is a reason why, I deliberately attach a GPT response, because maybe MAYBE... GPT-5 does not <i>reason<i> that well. And that is because this definition sounds *true* but is definitely not academically *true*. Let us also think step-by-step. <br> <br>

1. How does Human Reasoning differs compared to LLM Reasoning <br>
Humans use neural-symbolic reasoning: a mixture of intuition and explicit symbolic manipulation. The brain supports working memory, goal-directed planning, and causal models. We can apply rules, logic, and abstraction deliberately. <br>
LLMs perform next-token prediction using a statistical function learned from large corpora. All "reasoning" emerges from pattern completion, not explicit symbolic rules. LLMs do not maintain persistent working memory; they rely on the context window and hidden activations. 

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

### **System 1 vs System 2: Association vs Deliberation**
Borrowing from cognitive science, LLM behaviour mirrors System 1 thinking -- fast, associative, pattern based (recalling factual knowledge). <br>
True multi-step reasoning, analogous to System 2 deliberation, demands:
1. Representation of intermediate states (a mental scratchpad)
2. Logical dependency tracking
3. Dynamic allocation of computation proportional to task difficulty. 
Early Transformer models lacked these properties because their computation depth is fixed: every problem receives the same number of forward passes regardless of complexity. Thus, “thinking longer” was architecturally impossible.

### **The role of In-context Learning**
The discovery of in-context learning (ICL) revealed that sufficiently large models can learn new tasks purely from prompts—no parameter updates needed. However, ICL only becomes robust beyond certain scale thresholds (tens of billions of parameters) <br>
Smaller models can mimic patterns but fail to generalize relational structure across tasks. This observation seeded a core hypothesis repeated throughout reasoning research:
> Reasoning abilities emerge when model scale × data diversity × prompt structure exceed a critical threshold. <br><br>

Because LLMs are, by nature, static pattern engines, researchers sought to force structure into their thought process: to make the model externalize intermediate reasoning, reuse prior logic, and self-correct when diverging. This led to a family of methods that can be organized thematically. In the sections that follow, we will track this evolution thematically—from explicit reasoning traces to self-aware deliberation frameworks—analyzing not only what worked but why each innovation succeeded or failed, and what it reveals about the illusion (or emergence) of machine thought.

## **Part I: Explicit Intermediate Reasoning**
> Reasoning abilities are not emergent accidents; they are trained statistical habits of expressing thought.

### Show your work: Scratchpads (2021, Google Research - Brain Team)
![scratchpad](./imgs/scratchpads.png)
Before "Chain-of-Thought" existed, *Show your Work (Nye et al., 2021) tackled a simple question: 
> Can Transformers perform multi-step computation if we explicitly make them write intermediate steps?

The answer arrived through **Scratchpads**: texual traces inserted between input and output. Instead of forcing a single forward pass, the model learned to externalize its computation in human readable form. <br> <br>
**Method** <br>
- Each training example contained the **intermediate algorithmic trace** of a Python program. 
- The model predicted not only the final answer but also the full execution trace: 
  - The order of source lines executed
  - The state of local variables after each line. 
This process was called a *trace exact match* evaluation: semantic comparision of predicted and ground-truth traces + sequence alignment of executed lines.<br>
**Findings** <br>
Scratchpads improved performance on synthetic tasks but failed on MBPP (Mini Python Benchmark Programs). Why? <br>
- MBPP was too small (~ 400 trainin examples)
- Transformers are pattern matchers -- without ample scratchpad-style examples, they fail to generalize to the format (**RECALL IN CONTEXT LEARNING HERE**) <br>
To fix this, the authors built **MBPP-Aug** via data augmentation:
- A 137 B-parameter model generated ≈ 80 candidate programs per task at T = 0.5.
- Each program was executed on original inputs to filter failures.
- Execution traces were recorded for successful programs. <br>

**Result**: Scratchpads scaled successfully with dataset size — a clear hint that reasoning is a data-driven phenomenon. 

My take on this paper:
> Reasoning abilities are a result of large datasets having scratchpad-like examples, not just the emergent abilities of Transformer. 

Why?
- Smaller datasets limit the model's ability to infer the format of multi-step reasoning. 
- This invites a test: could Scratchpads be implemented on non-Transformer architectures (RNNs with external memory, diffusion models over traces)?

### Chain-of-Thought Prompting (2022, Google Research -- Brain Team)
<p align="center">
  <img src="./imgs/cot.png" width="400">
</p>
LLMs can "reason" if we make them explain their answers. **Chain-of-Thought (CoT)** prompting introduces a natural-language "reasoning-chain" between question and answer. <br>
Few-shot CoT requires only a handful of exemplars like:

``` vbnet
Q: If there are 3 cars and each has 4 wheels, how many wheels?
A: Each car has 4 wheels -> 3x4 = 12 -> 12. 
```
This seemingly simple pattern doubles performance on multi-step benchmarks. This taught models to narrate their reasoning. Large Models suddenly solved multi-step arithmetic and commonsense problems far better, while smaller ones produced nonsense chains that hurt accuracy. <br>
Manual insepction revealed something deeper: some mathematically wrong chains still produced correct answers by conincidence. *This hinted that LLMs were not truly reasoning; they were sampling statistically plausible stories of reasoning* <br>

Ablations clarified the mechanism:<br>
- **Equation-only prompting** helped for single-step math but failed for semantic tasks.
- **Variable-compute prompting** gave inconsistent results; length is not a good awat proxy for congitive effort.
- **Post-answer chains** offered no gain, showing that the process of reasoning, not just recalling facts, drives success. 

**Experiments & Datasets**
Models: GPT-3, LaMBDA, PaLM, UL2 20B, Codex <br>
Datasets: GSM8K, SVAMP, ASDiv, AQuA, MAWPS <br>
Authors hand-crafted 8 few-shot CoT exemplars covering arithematic and commonsense reasoning. <br> <br>

**Key Findings**
- CoT improves performance **only for large models (> 100B)**
- Smaller models produce illogical chains that hurt accuracy relative to standard prompting
- Gains are largest on multi-step tasks (GSM8K, StrategyQA); negligible on simple arithematic (SingleOp MAWPS) <br><br>
Manual inspection showed that some correct answers arose from **incorrect reasoning** (*lucky conincidence* right?). So maybe LLMs weren't yet "reasoners" -- they imitated the surface pattern of reasoning. <br>


### Zero-Shot CoT  (2022, Google Research)
Few-shot CoT needed human examples. <br>
Zero-Shot CoT eliminited them with the trigger phrase:
> Let's think step by step.

<br>
The model first generates a reasoning trace $$z$$, then re-prompts with the trace to obtain the final answer. This “double prompting” works astonishingly well for huge models like PaLM 540 B, but smaller ones barely benefit. Temperature sampling sometimes rescues a poor reasoning path—an accidental discovery that stochastic decoding performs a kind of search over possible thoughts. This can be interpreted as evidence that **reasoning = search** in text space. Temperature controls exploration breadth, letting the model stumble into valid logical sequences. Thus, even without curated exemplars, reasoning can be elicited linguistically if the model is large enough to internalize the pattern.

### What explicit reasoning taught us
From Scratchpads to Auto-CoT, a pattern emerges.
LLMs can reason only when we externalize reasoning as text and feed them examples of it.
Once the “format of thought” becomes part of the training distribution, they learn to reproduce it.

**Key Insights**:
- **Reasoning is data-driven**. Scale × diversity × explicit structure = ability.

- **Language is computation**. The model’s own text serves as its working memory.

- **Search beats static prompting**. Temperature, sampling, and automated demonstration selection act like exploration policies.

- **Faithfulness is the next frontier**. A correct answer isn’t enough; we must align the reasoning path with truth.

What I think about is, that LLMs are statistical reasoners, then teaching them to write down their statistical reasoning is how we make them look rational. Explicit intermediate reasoning does not make models think, it makes their stochastic prediction process legible. <br>

What follows in the next section, is how researchers pushed beyond single-pass reasoning toward *hierarchical decomposition* and *compositional thought*. 

## **Part II: Hierarchical & Compositional Reasoning**
> “Faithfulness increases when reasoning depth exceeds five steps.”

### **Least-to-Most Prompting (Google Research)**
If Chain-of-Thought describes what the model is doing, Least-to-Most Prompting (LtM) tells it how to plan. A complex question is decomposed into a sequence of simpler sub-questions, each solved in order. The model is first asked to split a task, then to solve each sub-task sequentially—mirroring how humans plan multi-step solutions. <br>

This is a bit intuitive. Think of an exam. 
```vbnet
Question 1: Solve the following questions:
Equation I: <some equation>
a) Compute Hessian Matrix of some equation I.
b) Is the matrix semi positive definitie? 
c) Comment on the mimina/maxima of the equation.
```
In the above question, you are forced to solve the over-arching question, *What is the minima and maxima of Equation I*, in sub-parts. Solving for Hessian first, then identifying the property of the matrix, and then finally commenting on the minima/maxima of the equation. This makes your job easy, right? Instead of jumping straight into minima/maxima, you solve the problem through some said-steps. This is what Least-to-Prompting does. Instead of directly jumping to the solution, you break down the problem into sub-parts, and solve them to reach the solution. <br>

**Implementation Outline**
1. Decomposition pass → “Break the problem into smaller parts.”
2. Solution pass → “Now solve them in order.”

Often both passes are combined into one long prompt for efficiency. <br>

**Results**
LtM consistently improved over CoT on symbolic reasoning (e.g., Last-Letter Concatenation) and on compositional generalization datasets such as SCAN, where sequences in the test set are longer than those in training. It reduced classic CoT failure modes—skipping intermediate facts, dropping letters, or concatenating incorrectly. However, 100 % accuracy remained elusive: models still made “off-by-one” logic errors or hallucinated extra symbols. The gain appeared only when reasoning depth exceeded roughly 5 steps, meaning LtM is most valuable for hard multi-step tasks.

**My thoughts**
- LtM improves faithfulness, the reasoning stays on-topic instead of drifting, 
- Its benefit is selective: trivial tasks waste compute with needless decomposition. 
- Evaluation should focus on reasoning depth, not dataset size; benchmarks need long-chain problems to reveal true differences. 
- Structured decomposition enables stable multi-step reasoning.
- Two-stage prompting acts like curriculum learning inside the model. 
- Effective only when the reasoning path is sufficiently long or entangled. 

### **Measuring the COmpositionality Gap (Meta AI Research)**
Just when you thought the reasoning sounds more like Human (contradictory to what I said above), this explains why even breaking down the problems does not always yield correct answers. How often do you solve all the parts correctly, but do not write the correct final answer? The chance of this happening is pretty low, unless you're very careless with calculator, or maybe your pencil broke? Or your cat ran over your keyboard? Well Large Language Models do often fail to combine their answers into the final answer. This mismatch defined the **compositionality gap**. <br>

**Experiment**: Meta AI built *Compositionality Celebrities:* questions combining two unrelated facts eg:
> Who won the Masters Tournament the year Justin Bieber was born? <br>
Each requires two hops -- (Beiber -> birth year, year -> winner). 

Larger models answered both sub-questions easily but frequently failed the joint query. Single-hop factual recall scaled fast; multi-hop composition did not. Thus, model size helps memorization (System 1) more than relational reasoning (System 2). <br>

**Closing the gap -- Self-Ask Prompting**
To patch this, researchers proposed **Self-Ask**, where the model explicitly asks itself follow-up questions before answering. 

```yaml
Q: Who won the Masters Tournament the year Justin Bieber was born?
A: Let's break this down.
Follow-up: When was Justin Bieber born?
Intermediate answer: 1994.
Follow-up: Who won the Masters in 1994?
Final answer: Jose Maria Olazabal.
```

When coupled with an external search tool, each follow up can query real data, forming a lightweight reasoning loop. This scaffold improved both accuracy and explanatory clarity without finetuning. Multi-hop reasoning reveals whether a model composes knowledge rather than merely recalls it. The compositionality metric separates knowing from thinking. Integrating Self-Ask with search engines shows that reasoning quality can rise without increasing parameters—better scaffolding beats bigger models. The gap widens with scale: factual recall ≠ reasoning. Prompt scaffolding (Self-Ask) narrows it by enforcing decomposition. True compositional reasoning demands controllable intermediate queries.

> Reasoning depth, not dataset length, predicts real understanding

Hierarchical and compositional approaches taught LLMs to structure their internal logic. Instead of producing a single narrative chain, they learned to build trees of sub-goals whose answers could be recombined. This transition—from linear explanation to compositional synthesis—set the stage for methods that reason by comparison, correction, and self-feedback, explored next. 

## **Part III: Self-Consistency & Self Correction**
> Sampling many lines of reasoning is like consulting many minds inside the same model. 

### **Self Consistency (Google, Research)**
When Chain-of-Thought was first tested, every example relied on a single reasoning path produced by greedy decoding. But language models are stochastic: a different temperature or random seed can yield an entirely new explanation. Self-Consistency (SC) embraced this randomness. Instead of forcing one “correct” chain, the model samples M reasoning traces, each concluding with an answer. The final prediction is then chosen by majority vote (or a probability-weighted vote) over these sampled completions. This technique is deceptively simple yet powerful. It replaces determinism with a distribution over thoughts, averaging out individual hallucinations. Because independent samples explore diverse reasoning routes, convergence among them acts as an internal reliability signal. <br>

**Observations**
- The method is **completely unsupervised**, no retrainined or labels.
- Mejority and weighted voting give nearly identical accuracy because LLMs assign similar normalized probabilities to coherent reasoning paths. 
- Self-Consistency is **robust** across decoding temperature, prompt orders, and model scales. 
- Gains are largest for multi-step arithetic and commonsense tasks (GSM8K, CSQA, StrategyQA). 

We may slighly hit off on a different road here, and actually see how this works mathematically. So feel free to skip this section if you want. <br>

**Derivation** <br>
Suppose you have a model, and you sample $$m$$ independent chain-of-thought traces $$ {\tau_{1}, \tau_{2}, ..., \tau_{m}} $$. Each trace $$\tau_{i}$$ ends in a final answer
$$
a_{i} \in A 
$$
(eg, integers, strings, multiple-choice labels). You want a final prediction by **majority vote over the answers $$ {a_1, ..., a_m}$$
<br>

**1.Define Answer Counts** <br>
For each unique answer $$ a \in A $$, define the vote count: <br>

$$
C(a) = \sum_{i=1}^{m} \mathbf{1}\{ a_i = a \}, \qquad 
\mathbf{1}\{ a_i = a \} =
\begin{cases}
1, & \text{if } a_i = a, \\
0, & \text{otherwise}.
\end{cases}
$$

**2. Define Majority Vote Decision Rule** <br>

The majority-vote estimator chooses the answer with the highest count:

$$
\hat{a} = \text{arg max}_{a \in A} C(a)
$$

Explicitly:

$$
\hat{a} = \text{arg max}_{a \ in A} \sum_{i=1}^m 1[a_i = a]
$$

This is the formal derivatin. 

**3. Tie-Breaking**
Many papers clarify tie-breaking because majority voting can yield ties when: 

$$
C(a) = C(b),  \qquad
\text{for some a } \neq \text{b}
$$

Typically tie breakers: <br>

- Random uniform tie-break:

$$
\hat{a} \sim Uniform ({a: C(a) = max_{a`}C(a`)})
$$






