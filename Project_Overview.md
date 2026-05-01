# Project Overview: Synthetic 5G Network Traffic Generation

This document provides a high-level overview of the two foundational papers and the proposed research task that bridges their concepts.

## 1. Overview of Paper 1
**Title:** A Real Network Environment Dataset for Traffic Analysis

**Summary:** 
This paper focuses on the capture of real-time network traffic and the subsequent generation of similar synthetic traffic using Artificial Intelligence (Generative AI). 
* **Core Concept:** Creating high-fidelity artificial network traffic that mimics the statistical and structural properties of real-world captures.
* **Validation:** The generated dataset is extensively analyzed and validated against the original real traffic using specific statistical metrics to prove its similarity and utility.

## 2. Overview of Paper 2
**Title:** 5G Wireless Network Intrusion Detection Dataset (5G-NIDD)

**Summary:** 
This paper introduces a modern, purpose-built dataset (5G-NIDD) generated from a real 5G wireless network environment.
* **Core Concept:** Providing a reliable benchmark for 5G-specific intrusion detection. It contains both normal benign traffic and various types of malicious attacks.
* **Validation:** The dataset's effectiveness is analyzed by training various Machine Learning (ML) models for Intrusion Detection Systems (IDS) and evaluating their detection accuracy on the 5G data.

---

## 3. Proposed Task

**Objective:** Synthesize 5G network traffic using Generative AI models and validate its fidelity.

The goal is to intersect the methodologies of both papers. We will take the highly specific **5G-NIDD dataset (Paper 2)** and treat it as our ground-truth data. We will then design and apply **Generative AI models (inspired by Paper 1)** to synthesize artificial network traffic that looks and behaves exactly like the 5G-NIDD data. 

### Key Steps to Perform:

1. **Data Preprocessing:**
   * Load and clean the 5G-NIDD dataset.
   * Handle the mixture of categorical (IPs, Ports, Protocols) and continuous features (packet sizes, flow durations, inter-arrival times).
   
2. **AI Model Selection & Training:**
   * Select appropriate Generative AI models. Options include tabular generators like **CTGAN (Conditional Tabular GAN)** or **TVAE**.
   * To handle the time-series nature of network flows, sequential models like **LSTMs (Long Short-Term Memory)** networks are highly suitable. LSTMs can be used either to sequentially predict the next packet characteristics or stacked inside GANs (e.g., **LSTM-GAN** or **TimeGAN**) to generate realistic sequences of traffic flows while preserving temporal context.
   * Train the models on both normal and attack traffic classes from the 5G-NIDD dataset.

3. **Traffic Synthesis:**
   * Generate new, synthetic 5G-NIDD flow records using the trained models.

4. **Validation and Metric Analysis:**
   * **Statistical Metrics:** Use tests like Kolmogorov-Smirnov (KS) and Jensen-Shannon (JS) divergence to prove the synthetic data columns match the real data distributions.
   * **Visual Metrics:** Plot PCA or t-SNE visualizations showing the overlap between real and synthetic data points.
   * **ML Utility Efficacy:** Train Intrusion Detection classifiers (from Paper 2) on the *synthetic* data and test them on the *real* 5G-NIDD dataset (Train on Synthetic, Test on Real - TSTR) to see if the AI successfully preserved the attack patterns.

### Potential Difficulties to Address:
* **Mode Collapse:** The generative model might struggle to generate the wide variety of rare 5G attacks, collapsing to only produce normal traffic or a single attack type. 
* *Mitigation:* Use **Class-Conditional Generation** (e.g., Conditional LSTMs or CGANs) to explicitly force the model to generate specific attack labels. Apply **WGAN-GP (Wasserstein GAN with Gradient Penalty)** which offers improved training stability over standard GAN loss functions. Additionally, using balanced batch sampling (showing the AI equal amounts of each attack type during training) prevents normal traffic from overpowering the network.
* **Temporal Dependencies:** Standard GANs struggle with the time-series nature of network flows. Preserving order and timing requires complex model tuning.
* **Complex Data Types:** Generating realistic IP addresses or constrained port numbers is mathematically difficult for standard neural networks.
