# System Architecture & Design Document: Synthetic 5G-NIDD Traffic Generation

**Document Version:** 1.0
**Role:** System Architecture Designer
**Project:** Generative AI for 5G Wireless Network Intrusion Detection Dataset (5G-NIDD)

---

## 1. Executive Summary
This document outlines the final system architecture for generating high-fidelity, synthetic 5G-NIDD network traffic. As a system designer, the objective is to build a robust pipeline that handles the heavy imbalance of network intrusion data, preserves the distinct time-series nature of 5G communications, and manages complex, mixed data types.

---

## 2. Final Proposed Architecture: Conditional TimeGAN (C-TimeGAN)

To address the complexities of time-series network flows, categorical attributes, and severe class imbalance, the recommended core architecture is a **Conditional Time-series Generative Adversarial Network (C-TimeGAN)**.

### 2.1 Pipeline Components
1. **Data Ingestion & Parser Module:** Reads the raw 5G-NIDD flow records (CSV/PCAP).
2. **Feature Engineering & Preprocessor:**
   * **Categorical Embedder:** Converts IPs, Ports, and Protocols into dense latent vectors rather than sparse one-hot encoding.
   * **Continuous Scaler:** Applies logarithmic scaling to heavy-tailed features like `Total_Bytes` or `Duration`.
   * **Sequence Generator:** Groups independent flows into chronological sequences (windows of size $N$) using `Inter-Arrival Time (IAT)`.
3. **The AI Engine (C-TimeGAN):**
   * **Embedding/Recovery Network:** An Autoencoder that maps the raw discrete/continuous features into a continuous latent space for stable GAN training.
   * **Generator (LSTM):** Takes random noise ($z$) AND a target class label ($c$) (e.g., "ICMP Flood"). Uses LSTM to generate a temporal sequence in the latent space.
   * **Discriminator (LSTM):** Evaluates if the sequence is real or fake, conditioned on the label ($c$).
   * **Supervisor Network (LSTM):** Ensures the step-wise temporal dynamics (transition from packet $t$ to packet $t+1$) strictly follow the real data distribution.
4. **Post-Processor & Rule Engine:** Converts generated latent variables back to realistic network constraints (e.g., rounding to nearest integer, enforcing port bounds, re-calculating absolute timestamps from delta-t).

---

## 3. Potential Difficulties & Architectural Mitigations

### 3.1 Mode Collapse & Severe Class Imbalance
* **Problem:** Normal traffic constitutes ~95% of the data. The generator will likely collapse and only produce normal traffic, failing to learn rare 5G-specific attacks.
* **Mitigation Strategy:**
  * **Conditional Generation:** Inject the class label ($c$) into every layer of the Generator and Discriminator. This forces the model to learn conditional distributions $P(X|C)$.
  * **WGAN-GP Loss:** Replace the standard JS-divergence loss with Wasserstein distance and a Gradient Penalty. This provides a smoother gradient for minority classes.
  * **Stratified Batching:** Ensure every training mini-batch contains an artificially balanced ratio of all attack types.

### 3.2 Complex Mixed Data Types
* **Problem:** Network data mixes highly skewed continuous variables (byte counts, durations) with distinct categorical ones (ports, protocols). Standard ML models generate continuous floating-point numbers, which makes no sense for a "Protocol" or "Port number".
* **Mitigation Strategy:**
  * **Gumbel-Softmax Activation:** For categorical outputs, use the Gumbel-Softmax trick in the generator. This allows backpropagation through categorical (discrete) choices.
  * **Entity Imbedding Space:** Train an Autoencoder first. The GAN never touches the raw ports or IP addresses; it only outputs and evaluates the *continuous latent embeddings*.

### 3.3 Preserving Temporal Dependencies
* **Problem:** Simple Tabular GANs treat row 10 and row 11 as unrelated. In 5G networks, a high-throughput burst or a DDoS attack relies entirely on the rapid succession of flows.
* **Mitigation Strategy:**
  * **Supervisor Step:** The TimeGAN architecture incorporates a Supervisor Network that explicitly penalizes the Generator if the transition from sequence index $T_1$ to $T_2$ violates the underlying probability of the real dataset.

---

## 4. Edge Cases & Failsafes

As a system designer, we must plan for scenarios where the model predicts theoretically impossible or anomalous values:

### Edge Case 1: Negative Time & Causality Violations
* **Scenario:** The neural network outputs a negative Inter-Arrival Time (IAT) or a negative packet length.
* **System Fix:** Do not predict absolute timestamps. Predict the *delta* ($\Delta T$) between flows. Apply a `ReLU` (Rectified Linear Unit) activation on the final output node for time and size features to strictly enforce non-negativity.

### Edge Case 2: Impossible Network Configurations
* **Scenario:** The AI generates a TCP flag combination that violates RFC standards (e.g., SYN+RST simultaneously) or port numbers > 65535.
* **System Fix:** Implement a **Deterministic Post-Processing Rule Engine**. Any generated value exceeding 65535 is clamped to max allowable ports. Invalid TCP flag combinations are masked out using bitwise logical gates over the generator's stochastic output.

### Edge Case 3: IP Address Hallucination
* **Scenario:** The AI generates valid-looking IP addresses that belong to completely different subnets, rendering the traffic syntactically valid but topologically impossible for the simulated 5G environment.
* **System Fix:** IP addresses should not be generated natively by the GAN. Generate *topological IDs* (e.g., "Host A", "Attacker B"). In the post-processor, map these IDs back to a strictly defined subnet dictionary matching the 5G-NIDD topology.

---

## 5. System Evaluation & Testing Pipeline
To ensure the generative architecture functions correctly natively, the system will execute an automated testing suite after generation:
1. **Sanity Checker:** Verifies 0% negative packet sizes and 100% valid Ports/Flags.
2. **Statistical Distance Suite:** Runs 2D PCA/t-SNE and calculates the KS-Test to ensure distribution overlap.
3. **ML Oracle Test:** Automatically trains an XGBoost classifier on the Synth data and measures the F1-score degradation against the Real 5G-NIDD test set.