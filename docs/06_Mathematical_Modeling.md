# Mathematical Modeling

[← Back to Dataset and Preprocessing](05_Dataset_and_Preprocessing.md) | [Next: Methodology →](07_Methodology.md)

---

## Table of Contents

1. [Problem Formulation](#1-problem-formulation)
2. [Transformer Architecture](#2-transformer-architecture)
3. [Loss Functions](#3-loss-functions)
4. [Governing Equations](#4-governing-equations)
5. [Integrated Gradients](#5-integrated-gradients)
6. [Optimization Dynamics](#6-optimization-dynamics)
7. [Statistical Model](#7-statistical-model)
8. [Clinical Symptom Scoring](#8-clinical-symptom-scoring)
9. [Crisis Detection Model](#9-crisis-detection-model)
10. [Summary Table](#10-summary-table)

---

## 1. Problem Formulation

### 1.1 Binary Classification Framework

**Task:** Map input text $x \in \mathcal{X}$ to binary label $y \in \{0, 1\}$

- $y = 0$: Control (no depression)
- $y = 1$: Depression detected

**Model:** Parameterized function $f_\theta: \mathbb{R}^{n \times d} \to [0, 1]$

$$
f_\theta(x) = P(y = 1 \mid x; \theta)
$$

Where:
- $x \in \mathbb{R}^{n \times d}$: Tokenized text (sequence length $n$, embedding dimension $d$)
- $\theta$: Model parameters (weights and biases)
- $f_\theta(x)$: Probability of depression class

**Decision Rule:**

$$
\hat{y} = \begin{cases}
1 & \text{if } f_\theta(x) \geq \tau \\
0 & \text{if } f_\theta(x) < \tau
\end{cases}
$$

Default threshold: $\tau = 0.5$

### 1.2 Input Representation

**Tokenization:** Text $\rightarrow$ Token sequence

$$
x = [w_1, w_2, \ldots, w_n]
$$

**Embedding Layer:** Tokens $\rightarrow$ Dense vectors

$$
E = [e_1, e_2, \ldots, e_n] \in \mathbb{R}^{n \times d}
$$

Where $e_i = \text{Embed}(w_i) \in \mathbb{R}^d$ (typically $d = 768$ for BERT-base)

**Special Tokens:**

$$
E = [\text{[CLS]}, e_1, e_2, \ldots, e_n, \text{[SEP]}]
$$

- `[CLS]`: Classification token (used for final prediction)
- `[SEP]`: Separator token (marks sequence boundary)

### 1.3 Dataset Distribution

**Training Set:** $\mathcal{D}_{\text{train}} = \{(x_i, y_i)\}_{i=1}^{N_{\text{train}}}$

**Class Distribution:**

$$
P(y = 0) = \frac{N_0}{N_{\text{train}}} = 0.54 \quad \text{(Control)}
$$

$$
P(y = 1) = \frac{N_1}{N_{\text{train}}} = 0.46 \quad \text{(Depression)}
$$

Where:
- $N_0 = 432$ (control samples)
- $N_1 = 368$ (depression samples)
- $N_{\text{train}} = 800$ (total training samples)

---

## 2. Transformer Architecture

### 2.1 Self-Attention Mechanism

**Query-Key-Value Computation:**

$$
Q = E W_Q, \quad K = E W_K, \quad V = E W_V
$$

Where:
- $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$: Learnable projection matrices
- $d_k = d / h$: Head dimension ($h = 12$ heads for BERT-base)

**Scaled Dot-Product Attention:**

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

**Attention Weights:**

$$
A_{ij} = \frac{\exp\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right)}{\sum_{j'=1}^n \exp\left(\frac{q_i \cdot k_{j'}}{\sqrt{d_k}}\right)}
$$

Where:
- $A_{ij}$: Attention weight from token $i$ to token $j$
- $q_i, k_j$: Query and key vectors
- Normalization factor $\sqrt{d_k}$ prevents gradient vanishing

**Properties:**
1. $\sum_{j=1}^n A_{ij} = 1$ (attention weights sum to 1)
2. $A_{ij} \geq 0$ (non-negative weights)
3. $A_{ij}$ measures semantic similarity between tokens $i$ and $j$

### 2.2 Multi-Head Attention

**Parallel Attention Heads:**

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O
$$

Where:

$$
\text{head}_i = \text{Attention}(Q W_Q^i, K W_K^i, V W_V^i)
$$

- $h = 12$: Number of attention heads
- $W_O \in \mathbb{R}^{d \times d}$: Output projection matrix

**Advantages:**
- Captures different types of relationships (syntax, semantics, co-reference)
- Each head specializes in different linguistic patterns
- Improves model expressiveness

### 2.3 Transformer Layer

**Single Layer Computation:**

$$
E^{(\ell+1)} = \text{LayerNorm}(E^{(\ell)} + \text{MultiHead}(E^{(\ell)}))
$$

$$
E^{(\ell+1)} = \text{LayerNorm}(E^{(\ell+1)} + \text{FFN}(E^{(\ell+1)}))
$$

**Feed-Forward Network (FFN):**

$$
\text{FFN}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2
$$

Where:
- $W_1 \in \mathbb{R}^{d \times 4d}$: Expansion matrix (768 → 3072)
- $W_2 \in \mathbb{R}^{4d \times d}$: Compression matrix (3072 → 768)
- Intermediate dimension: $4d = 3072$

**Layer Normalization:**

$$
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

Where:
- $\mu = \frac{1}{d}\sum_{i=1}^d x_i$: Mean
- $\sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$: Variance
- $\gamma, \beta$: Learnable scale and shift parameters
- $\epsilon = 10^{-12}$: Numerical stability constant

### 2.4 Full Model Stack

**BERT/RoBERTa Architecture:**

$$
E^{(0)} = \text{Embed}(x) + \text{PositionalEncoding}(x)
$$

$$
E^{(\ell)} = \text{TransformerLayer}(E^{(\ell-1)}) \quad \text{for } \ell = 1, \ldots, L
$$

$$
h = E^{(L)}[0, :] \quad \text{(extract [CLS] token)}
$$

$$
z = \text{Dropout}(h)
$$

$$
\text{logits} = z W_{\text{cls}} + b_{\text{cls}}
$$

$$
P(y = 1 \mid x) = \sigma(\text{logits}_1) = \frac{1}{1 + e^{-\text{logits}_1}}
$$

Where:
- $L = 12$: Number of transformer layers
- $h \in \mathbb{R}^{768}$: [CLS] representation
- $W_{\text{cls}} \in \mathbb{R}^{768 \times 2}$: Classification weights
- $\sigma$: Sigmoid activation

**Parameter Count (BERT-base):**
- Embedding layer: 23.4M parameters
- 12 Transformer layers: 85.0M parameters
- Classification head: 1,536 parameters
- **Total: 110M parameters**

---

## 3. Loss Functions

### 3.1 Binary Cross-Entropy Loss

**Objective Function:**

$$
\mathcal{L}_{\text{BCE}}(\theta) = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \log f_\theta(x_i) + (1 - y_i) \log(1 - f_\theta(x_i)) \right]
$$

Where:
- $N$: Batch size
- $y_i \in \{0, 1\}$: True label
- $f_\theta(x_i) \in [0, 1]$: Predicted probability

**Properties:**
- Convex in logit space
- Penalizes confident wrong predictions heavily
- $\mathcal{L} \to \infty$ as $f_\theta(x) \to 1 - y$

### 3.2 Weighted Cross-Entropy Loss

**Class Imbalance Handling:**

$$
\mathcal{L}_{\text{weighted}}(\theta) = -\frac{1}{N}\sum_{i=1}^N \left[ w_1 \cdot y_i \log f_\theta(x_i) + w_0 \cdot (1 - y_i) \log(1 - f_\theta(x_i)) \right]
$$

**Class Weights:**

$$
w_0 = \frac{N}{2 N_0} = \frac{800}{2 \times 432} = 0.926 \quad \text{(Control)}
$$

$$
w_1 = \frac{N}{2 N_1} = \frac{800}{2 \times 368} = 1.087 \quad \text{(Depression)}
$$

**Interpretation:**
- Depression class weighted 17% higher than control
- Compensates for 54/46 class imbalance
- Prevents bias toward majority class

### 3.3 Regularization Terms

**L2 Regularization (Weight Decay):**

$$
\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}_{\text{weighted}}(\theta) + \frac{\lambda}{2} \|\theta\|_2^2
$$

Where:
- $\lambda = 0.01$: Regularization strength
- $\|\theta\|_2^2 = \sum_{i} \theta_i^2$: Squared L2 norm

**Purpose:**
- Prevents overfitting
- Encourages smaller weights
- Improves generalization

**Total Loss:**

$$
\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{weighted}}(\theta) + \lambda \|\theta\|_2^2
$$

---

## 4. Governing Equations

### 4.1 Core Problem Formulation

**State Space Dynamics:**

$$
\frac{dS}{dt} = F(S, \theta, C)
$$

Where:
- $S(t) \in \mathbb{R}^{768}$: Hidden state (embedding space)
- $\theta$: Model parameters
- $C$: Contextual information (attention, position)
- $F$: Transition function (transformer layers)

**Classification Boundary:**

$$
g(S) = W^T \phi(S) + b = 0
$$

Where:
- $\phi(S)$: Feature transformation (final [CLS] representation)
- $W \in \mathbb{R}^{768}$: Decision hyperplane normal vector
- $b$: Bias term

**Region Classification:**

$$
y = \begin{cases}
1 & \text{if } g(S) > 0 \quad \text{(Depression)} \\
0 & \text{if } g(S) \leq 0 \quad \text{(Control)}
\end{cases}
$$

### 4.2 Attention Flow Equations

**Multi-Head Attention Flow:**

$$
A_{ij}^{(h)} = \frac{\exp\left(\frac{q_i^{(h)} \cdot k_j^{(h)}}{\sqrt{d_k}}\right)}{\sum_{j'=1}^n \exp\left(\frac{q_i^{(h)} \cdot k_{j'}^{(h)}}{\sqrt{d_k}}\right)}
$$

**Attention Rollout (Layer Aggregation):**

$$
A^{\text{rollout}} = \prod_{\ell=1}^L \bar{A}^{(\ell)}
$$

Where:

$$
\bar{A}^{(\ell)} = \frac{1}{h} \sum_{i=1}^h A^{(\ell, i)} + I
$$

- $A^{(\ell, i)}$: Attention matrix for layer $\ell$, head $i$
- $I$: Identity matrix (residual connection)

**Conservation Property:**

$$
\sum_{j=1}^n A_{ij} = 1 \quad \forall i
$$

### 4.3 Gradient Flow Equations

**Backpropagation Dynamics:**

$$
\frac{\partial \mathcal{L}}{\partial E^{(\ell)}} = \frac{\partial \mathcal{L}}{\partial E^{(\ell+1)}} \cdot \frac{\partial E^{(\ell+1)}}{\partial E^{(\ell)}}
$$

**Residual Connection Gradient:**

$$
\frac{\partial E^{(\ell+1)}}{\partial E^{(\ell)}} = I + \frac{\partial \text{Attn}(E^{(\ell)})}{\partial E^{(\ell)}}
$$

**Gradient Norm (Stability Metric):**

$$
\|\nabla_\theta \mathcal{L}\|_2 = \sqrt{\sum_i \left(\frac{\partial \mathcal{L}}{\partial \theta_i}\right)^2}
$$

**Exploding Gradient Prevention:**

Gradient clipping:

$$
\nabla_\theta \mathcal{L} \leftarrow \begin{cases}
\nabla_\theta \mathcal{L} & \text{if } \|\nabla_\theta \mathcal{L}\|_2 \leq c \\
c \cdot \frac{\nabla_\theta \mathcal{L}}{\|\nabla_\theta \mathcal{L}\|_2} & \text{otherwise}
\end{cases}
$$

Where $c = 1.0$ (max norm threshold)

### 4.4 Information Flow Dynamics

**Shannon Entropy of Attention:**

$$
H(A) = -\sum_{i=1}^n \sum_{j=1}^n A_{ij} \log A_{ij}
$$

**Information Bottleneck:**

$$
\min_\theta \mathcal{L}(y, \hat{y}) \quad \text{subject to} \quad I(X; Z) \geq I_{\min}
$$

Where:
- $I(X; Z)$: Mutual information between input $X$ and representation $Z$
- $I_{\min}$: Minimum information threshold

---

## 5. Integrated Gradients

### 5.1 Attribution Method

**Goal:** Attribute prediction to input features (tokens)

**Path Integral Formulation:**

$$
\text{IG}_i(x) = (x_i - x'_i) \times \int_{\alpha=0}^1 \frac{\partial f(x' + \alpha(x - x'))}{\partial x_i} d\alpha
$$

Where:
- $x$: Input text (original)
- $x'$: Baseline (zero embedding)
- $x_i$: $i$-th token embedding
- $\alpha \in [0, 1]$: Interpolation parameter
- $f$: Model output (logit for depression class)

**Interpretation:**
- Measures contribution of token $i$ to prediction
- Integrates gradients along path from baseline to input
- $\text{IG}_i > 0$: Token increases depression probability
- $\text{IG}_i < 0$: Token decreases depression probability

### 5.2 Axioms

**1. Completeness:**

$$
f(x) - f(x') = \sum_{i=1}^n \text{IG}_i(x)
$$

Prediction difference equals sum of attributions.

**2. Sensitivity:**

If $x_i \neq x'_i$ and $f(x) \neq f(x')$, then $\text{IG}_i(x) \neq 0$

Changed features get non-zero attribution.

**3. Implementation Invariance:**

Two functionally equivalent networks produce identical attributions.

### 5.3 Riemann Approximation

**Discrete Sum (Practical Implementation):**

$$
\text{IG}_i(x) \approx (x_i - x'_i) \times \sum_{k=1}^m \frac{\partial f(x' + \frac{k}{m}(x - x'))}{\partial x_i} \cdot \frac{1}{m}
$$

Where $m = 20$ steps (default in Captum library)

**Convergence Property:**

$$
\lim_{m \to \infty} \sum_{k=1}^m \frac{\partial f}{\partial x_i}\Big|_{x' + \frac{k}{m}(x - x')} \cdot \frac{1}{m} = \int_{\alpha=0}^1 \frac{\partial f(x' + \alpha(x - x'))}{\partial x_i} d\alpha
$$

**Error Bound:**

$$
\epsilon(m) \leq \frac{M}{2m} \cdot \|x - x'\|_\infty
$$

Where:
- $M$: Maximum second derivative of $f$
- $m$: Number of steps
- For $m = 20$, approximation error $< 5\%$

### 5.4 Token-Level Attribution

**Aggregate Subword Tokens:**

$$
\text{IG}_{\text{word}} = \sum_{j \in \text{subwords}} \text{IG}_j
$$

**Example:**
- Token: "worthless" → Subwords: ["worth", "##less"]
- $\text{IG}_{\text{worthless}} = \text{IG}_{\text{worth}} + \text{IG}_{\text{\#\#less}}$

**Normalization:**

$$
\text{IG}'_i = \frac{\text{IG}_i}{\max_j |\text{IG}_j|}
$$

Result: Normalized scores in $[-1, 1]$

### 5.5 Visualization Mapping

**Color Coding:**

$$
\text{color}_i = \begin{cases}
\text{red} & \text{if } \text{IG}'_i > 0.5 \quad \text{(strong positive)} \\
\text{orange} & \text{if } 0.2 < \text{IG}'_i \leq 0.5 \\
\text{yellow} & \text{if } 0 < \text{IG}'_i \leq 0.2 \\
\text{gray} & \text{if } \text{IG}'_i \leq 0
\end{cases}
$$

**Top-K Selection:**

$$
\text{Top-K} = \arg\text{topk}_{i=1,\ldots,n}(\text{IG}_i, k=10)
$$

Highlight 10 most important words.

---

## 6. Optimization Dynamics

### 6.1 AdamW Optimizer

**Update Rule:**

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \lambda \eta \theta_t
$$

**Momentum (First Moment):**

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

**Variance (Second Moment):**

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

**Bias Correction:**

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

**Parameters:**
- $\eta = 2 \times 10^{-5}$: Learning rate
- $\beta_1 = 0.9$: Momentum coefficient
- $\beta_2 = 0.999$: Variance coefficient
- $\epsilon = 10^{-8}$: Numerical stability
- $\lambda = 0.01$: Weight decay
- $g_t = \nabla_\theta \mathcal{L}(\theta_t)$: Gradient at step $t$

### 6.2 Learning Rate Schedule

**Warmup Phase:**

$$
\eta_t = \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}} \quad \text{for } t \leq T_{\text{warmup}}
$$

**Linear Decay:**

$$
\eta_t = \eta_{\max} \cdot \frac{T_{\text{total}} - t}{T_{\text{total}} - T_{\text{warmup}}} \quad \text{for } t > T_{\text{warmup}}
$$

Where:
- $T_{\text{warmup}} = 100$ steps
- $T_{\text{total}} = 2400$ steps (3 epochs × 800 samples)
- $\eta_{\max} = 2 \times 10^{-5}$

### 6.3 Convergence Analysis

**Convergence Rate (Strongly Convex Case):**

$$
\|\theta_t - \theta^*\| \leq (1 - \mu \eta)^t \|\theta_0 - \theta^*\|
$$

Where:
- $\theta^*$: Optimal parameters
- $\mu$: Strong convexity constant
- Exponential convergence

**Generalization Bound:**

$$
\mathbb{E}[\mathcal{L}_{\text{test}}] \leq \mathcal{L}_{\text{train}} + O\left(\sqrt{\frac{\log(1/\delta)}{N}}\right)
$$

With probability $1 - \delta$, test loss bounded by training loss + complexity term.

---

## 7. Statistical Model

### 7.1 Bayesian Formulation

**Posterior Probability:**

$$
P(y = 1 \mid x) = \frac{P(x \mid y = 1) P(y = 1)}{P(x \mid y = 1) P(y = 1) + P(x \mid y = 0) P(y = 0)}
$$

**Likelihood Ratio:**

$$
\log \frac{P(x \mid y = 1)}{P(x \mid y = 0)} \approx f_\theta(x)
$$

Neural network approximates log-likelihood ratio.

### 7.2 Calibration

**Temperature Scaling:**

$$
P_{\text{cal}}(y = 1 \mid x) = \frac{\exp(f_\theta(x) / T)}{\exp(f_\theta(x) / T) + \exp(0)}
$$

Where $T > 0$ is temperature parameter (typically $T \approx 1.5$)

**Expected Calibration Error (ECE):**

$$
\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
$$

Where:
- $B_m$: Samples in confidence bin $m$
- $\text{acc}(B_m)$: Accuracy in bin $m$
- $\text{conf}(B_m)$: Average confidence in bin $m$
- Target: ECE $< 0.05$

### 7.3 Uncertainty Quantification

**Epistemic Uncertainty (Model Uncertainty):**

$$
\mathbb{V}[P(y \mid x)] = \frac{1}{K} \sum_{k=1}^K P_k(y \mid x)^2 - \left(\frac{1}{K} \sum_{k=1}^K P_k(y \mid x)\right)^2
$$

Using Monte Carlo Dropout with $K = 10$ forward passes.

**Aleatoric Uncertainty (Data Uncertainty):**

$$
H(y \mid x) = -\sum_{c=0}^1 P(y = c \mid x) \log P(y = c \mid x)
$$

Shannon entropy of predictive distribution.

---

## 8. Clinical Symptom Scoring

### 8.1 PHQ-9 Mathematical Model

**Severity Score:**

$$
S = \sum_{j=1}^9 s_j \cdot w_j
$$

Where:
- $s_j \in \{0, 1, 2, 3\}$: Symptom frequency score
  - 0: Not at all
  - 1: Several days
  - 2: More than half the days
  - 3: Nearly every day
- $w_j$: Weight for symptom $j$ (default $w_j = 1$)

**Text-to-Score Mapping:**

$$
s_j = \mathbb{1}_{\text{symptom}_j}(x) \cdot \text{intensity}(x)
$$

Where:
- $\mathbb{1}_{\text{symptom}_j}(x)$: Binary indicator (symptom present)
- $\text{intensity}(x) \in \{1, 2, 3\}$: Inferred from modifiers ("always" → 3, "often" → 2, "sometimes" → 1)

**Composite Score:**

$$
S = \sum_{j=1}^9 \mathbb{1}_{\text{symptom}_j}(x) \cdot \text{intensity}_j
$$

### 8.2 DSM-5 Criteria

**Major Depressive Episode:**

$$
\text{MDE} = \left(\sum_{j=1}^9 \mathbb{1}_{\text{symptom}_j} \geq 5\right) \land (\mathbb{1}_{\text{depressed}} \lor \mathbb{1}_{\text{anhedonia}})
$$

**Constraints:**
1. At least 5 of 9 symptoms present
2. Must include either depressed mood OR anhedonia
3. Duration $\geq$ 2 weeks: $t \geq 14$ days

---

## 9. Crisis Detection Model

### 9.1 Risk Scoring

**Keyword-Based Risk Score:**

$$
R_{\text{crisis}} = \sum_{k=1}^{K} w_k \cdot \mathbb{1}_{\text{keyword}_k}(x)
$$

Where:
- $K = 100+$: Number of crisis keywords
- $w_k \in [0.6, 1.0]$: Keyword weight
  - High-risk: $w_k = 1.0$ (e.g., "suicide", "kill myself")
  - Medium-risk: $w_k = 0.6$ (e.g., "hopeless", "no point")

**Decision Rule:**

$$
\text{CRISIS} = \begin{cases}
\text{True} & \text{if } R_{\text{crisis}} > \tau_{\text{crisis}} \\
\text{False} & \text{otherwise}
\end{cases}
$$

Where $\tau_{\text{crisis}} = 0.8$ (threshold)

### 9.2 Multi-Factor Assessment

**Intent Score:**

$$
R_{\text{intent}} = \max_{p \in \text{patterns}} w_p \cdot \mathbb{1}_p(x)
$$

Patterns:
- "I will..." → $w = 0.9$
- "I'm going to..." → $w = 0.9$
- "I plan to..." → $w = 0.8$
- "I want to..." → $w = 0.6$

**Plan Specificity Score:**

$$
R_{\text{plan}} = \min\left(1.0, \sum_{i \in \text{indicators}} 0.3 \cdot \mathbb{1}_i(x)\right)
$$

Indicators: method (gun, pills), timing (tonight, tomorrow), preparation (note, goodbye)

**Combined Risk:**

$$
R_{\text{total}} = \alpha R_{\text{crisis}} + \beta R_{\text{intent}} + \gamma R_{\text{plan}}
$$

Weights: $\alpha = 0.5$, $\beta = 0.3$, $\gamma = 0.2$ (sum = 1.0)

---

## 10. Summary Table

| **Equation** | **Purpose** | **Key Parameters** |
|--------------|-------------|--------------------|
| $f_\theta(x) = P(y=1 \mid x)$ | Binary classification | $\theta$ (110M params) |
| $A_{ij} = \text{softmax}(q_i \cdot k_j / \sqrt{d_k})$ | Attention mechanism | $d_k = 64$ |
| $\mathcal{L} = -\sum [y \log \hat{y} + (1-y) \log(1-\hat{y})]$ | Loss function | Class weights: 0.93, 1.09 |
| $\text{IG}_i = (x_i - x'_i) \int_0^1 \frac{\partial f}{\partial x_i} d\alpha$ | Token attribution | $m = 20$ steps |
| $A^{\text{rollout}} = \prod_\ell \bar{A}^{(\ell)}$ | Attention flow | $L = 12$ layers |
| $\theta_{t+1} = \theta_t - \eta \hat{m}_t / \sqrt{\hat{v}_t} - \lambda\eta\theta_t$ | AdamW optimizer | $\eta = 2 \times 10^{-5}$, $\lambda = 0.01$ |
| $P_{\text{cal}}(y \mid x) = \exp(f/T) / Z$ | Confidence calibration | $T \approx 1.5$ |
| $\mathbb{V}[P(y \mid x)]$ | Epistemic uncertainty | MC Dropout, $K = 10$ |
| $S = \sum_{j=1}^9 s_j w_j$ | PHQ-9 scoring | $s_j \in \{0, 1, 2, 3\}$ |
| $R_{\text{crisis}} = \sum w_k \mathbb{1}_{\text{keyword}_k}$ | Crisis detection | $\tau = 0.8$ |
| $\text{ECE} = \sum \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$ | Calibration metric | Target: $< 0.05$ |

---

## Key Insights

1. **Transformer Architecture:** 12-layer BERT/RoBERTa with 110M parameters captures complex linguistic patterns
2. **Attention Flow:** Multi-head attention with rollout mechanism tracks information propagation
3. **Integrated Gradients:** Path integral method with completeness guarantee attributes predictions to tokens
4. **Optimization:** AdamW with warmup + linear decay achieves stable convergence
5. **Calibration:** Temperature scaling reduces overconfidence (ECE < 0.05)
6. **Clinical Scoring:** PHQ-9 mathematical model maps text to severity scores
7. **Crisis Detection:** Multi-factor risk assessment with 80% threshold

---

[← Back to Dataset and Preprocessing](05_Dataset_and_Preprocessing.md) | [Next: Methodology →](07_Methodology.md)
