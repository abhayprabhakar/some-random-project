# Experiment Change Log

This document records the code changes made during the final experiment cycle and the measured impact on training and synthetic-data quality.

## What changed

### Training pipeline
- Reduced the discriminator update ratio from `n_critic=3` to `n_critic=1` in `src/models/train.py`.
- Kept cuDNN RNNs disabled so double-backward GAN training remains stable on GPU.
- Added feature-matching loss from discriminator intermediate features to the generator objective.
- Added logging for the feature-matching term (`g_fm_loss`) in `checkpoints/training_metrics.csv`.

### Discriminator
- Added a `features()` method in `src/models/discriminator.py` to expose intermediate RNN activations for feature matching.

### Generator / sampling
- Kept label embedding-based conditioning in `src/models/generator.py`.
- Updated `src/models/generate.py` to prefer `checkpoints/model_final.pt` and fall back to `latest_checkpoint.pt`.
- Kept synthetic label sampling aligned to the real class prior for the final generation run.

## What was observed

### Training behavior improved
- The run completed cleanly and saved `checkpoints/model_final.pt`.
- Autoencoder reconstruction became much stronger during the long training run, with AE loss dropping to roughly `0.0139` at epoch 110 in the earlier baseline run.
- In the shorter feature-matching run, the discriminator/generator reached a more stable equilibrium and `g_fm_loss` dropped to near zero by the end.

### Synthetic fidelity did not improve
- Final evaluation on the new synthetic samples produced worse quality metrics than the earlier baseline.
- New run metrics:
  - KS distance: `0.9124`
  - JS divergence: `0.4565`
  - TSTR weighted F1: `0.0354`
- Earlier baseline was better on all three metrics, so the refinement improved optimization behavior but not distributional fidelity.

## Conclusion

The experiment is useful as an ablation result: it shows that adding feature matching made training look cleaner, but it did not translate into better synthetic data. The stronger baseline remains the better result to report, while this run should be documented as a negative or neutral experiment.

## Recommended next step

If further improvement is needed, the next sensible change is stronger conditioning or a different sequential generator family rather than more epochs.