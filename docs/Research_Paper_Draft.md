# Synthesizing High-Fidelity 5G Network Intrusion Traffic via Conditional Time-series Generative Adversarial Networks

## Abstract
The rapid deployment of 5th Generation (5G) cellular networks has introduced novel architectures such as Multi-Access Edge Computing (MEC), necessitating robust security mechanisms. While datasets like the 5G Wireless Network Intrusion Detection Dataset (5G-NIDD) provide crucial benchmarks for training Intrusion Detection Systems (IDS), the severe class imbalance and static nature of empirical datasets limit iterative ML development. Drawing upon methodologies for AI-driven traffic synthesis, this paper proposes a Conditional Time-series Generative Adversarial Network (C-TimeGAN) framework to synthesize high-fidelity, sequence-aware 5G network traffic. By embedding discrete and continuous flow features into a continuous latent space and employing Long Short-Term Memory (LSTM) adversarial networks, the proposed model learns temporal correlations and conditional attack distributions, successfully overcoming mode collapse and categorical restrictions.

## 1. Introduction
Modern 5G networks exhibit high throughput, ultra-low latency, and massive device density, creating complex network flow patterns that standard Generative Adversarial Networks (GANs) struggle to replicate. Empirical captures, such as those provided by the 5G-NIDD, offer a foundation for analysis but inherently suffer from severe class imbalances—benign traffic overwhelms malicious samples, leading to biased IDS models. 
This research bridges the gap by leveraging Generative AI to synthesize artificial traffic that mimics the exact statistical and structural sequences of real 5G environments.

## 2. Dataset Preprocessing
The foundational dataset utilized is the **5G-NIDD (Combined.csv)**, comprosing over 1.2 million network flow records across 52 features.

### 2.1. Feature Normalization and Encoding
Network flows contain mixed data types. The framework partitions features into:
* **Continuous Features ($X_{cont}$):** Variables such as `Duration`, `TotBytes`, and `Rate` exhibit heavy-tailed distributions. These are normalized via MinMax Scaling to bound limits within $[0, 1]$, preventing gradient explosion during backpropagation.
* **Categorical Features ($X_{cat}$):** Variables such as `Protocol`, `State`, and `Attack Type` are label-encoded integers. 

### 2.2. Sequence Windowing
As network attacks (e.g., DoS, Port Scans) possess strict temporal dependencies, tabular rows are converted into 3D tensors. A sliding window mechanism constructs sequences $S = \{x_t, x_{t+1}, ..., x_{t+N}\}$, where $N = 20$ timesteps per sequence, grouped by inter-arrival times.

## 3. Proposed Architecture: C-TimeGAN
The architecture resolves the mathematical constraints of generating discrete variables within sequenced data by separating the mapping phase from the generative phase.

### 3.1. Latent Space Representation (Autoencoder)
Applying GANs directly to discrete integer values (like Port Numbers) breaks differentiability. Therefore, an **LSTM Autoencoder** is trained first. The Encoder network compresses the 51-dimensional flow vector at time $t$ into a smaller continuous $D$-dimensional latent vector $h_t$. The Decoder reconstructs $x_t$ from $h_t$. The subsequent GAN strictly operates within this continuous latent space $H$.

### 3.2. Conditional Adversarial Training
The system employs a **Conditional GAN (CGAN)** structure.
* **Condition ($C$):** The specific `Attack Type` label is passed through a dense embedding layer and concatenated to the inputs of both the Generator and Discriminator. This forces the model to learn the distribution $P(X|C)$, preventing mode collapse into the majority "Benign" class.
* **Generator ($G$):** An LSTM network taking random Gaussian noise $Z_t$ and condition $C$, emitting a fake latent sequence $\hat{h}_{1:T}$.
* **Discriminator ($D$):** An LSTM network that receives sequences $h_{1:T}$ and labels $C$, trained to distinguish empirical embeddings from generated ones.

To stabilize training, the framework replaces standard JS-Divergence loss with **Wasserstein Loss with Gradient Penalty (WGAN-GP)**.

## 4. Evaluation Methodology
To validate structural fidelity and utility, the generated data undergoes rigid assessment.

### 4.1. Statistical and Distributional Distance
* **Kolmogorov-Smirnov (KS) Test:** Measures the maximum distance between the empirical Cumulative Distribution Functions (CDFs) of real and synthetic continuous features.
* **Jensen-Shannon (JS) Divergence:** Evaluates the probability overlap of discrete categorical feature histograms.

### 4.2. Visual Dimensionality Reduction
Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE) project the 51-dimensional data into standard 2D Euclidean space. Overlapping clusters between empirical and synthetic data indicate high distributional fidelity.

### 4.3. Machine Learning Efficacy (TSTR)
The ultimate validation uses the **Train on Synthetic, Test on Real (TSTR)** paradigm. A Random Forest classifier is trained entirely on the C-TimeGAN generated data. Its performance is then validated against the withheld, empirical 5G-NIDD test set. High precision, recall, and F1-scores confirm the generated traffic maintains actionable, rule-based representations of 5G cyberattacks.

## 5. Conclusion
By integrating sequential Autoencoders with Conditional LSTM-GANs, this framework enables the robust synthesis of 5G network flows, offering scalable, private, and balanced datasets for future telecommunications security research.
