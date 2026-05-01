# Synthetic 5G-NIDD Traffic Generator (C-TimeGAN)

## Overview
This project applies Generative AI to synthesize artificial 5G network traffic, utilizing the 5G Wireless Network Intrusion Detection Dataset (5G-NIDD) as ground truth. The system generates high-fidelity, time-series network flows that mimic real-world 5G characteristics and complex cyberattacks without suffering from mode collapse.

## Documentation
New developers should read the following documents in order:
1. `Project_Overview.md` - Context, goals, and metrics.
2. `System_Architecture_Design.md` - Detailed architectural choices, edge cases, and design patterns.
3. `Project_Tracker.md` - Current status of development.

## Directory Structure
```text
nitk-internship-project/
├── configs/              # Configuration files (YAML/JSON) for hyperparameters
├── data/
│   ├── raw/              # Raw 5G-NIDD dataset (Never modified)
│   ├── processed/        # Cleaned and sequence-mapped data
│   └── synthetic/        # Generated outputs from the AI model
├── docs/                 # Extended documentation (Overview, Design)
├── src/
│   ├── preprocessing/    # Data ingestion, scaling, and sequence grouping
│   ├── models/           # C-TimeGAN architecture (Autoencoder, Generator, Discriminator)
│   ├── postprocessing/   # Rule engines for fixing edge anomalies
│   └── evaluation/       # Statistical and ML metrics (PCA, KS-test, XGBoost Oracle)
├── notebooks/            # Jupyter notebooks for visual EDA and testing
├── Project_Tracker.md    # Development tracking
├── README.md             # Developer entry point
```

## Getting Started (WIP)
*Environment setup and dependency map will be updated as the pipeline is formulated.*
