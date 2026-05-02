# Changes and Results

This file summarizes the final experiment cycle for the 5G-NIDD synthetic traffic project.

## Changes Made

- Added stronger class conditioning in the generator and discriminator.
- Added a discriminator `features()` method to expose intermediate activations.
- Added feature-matching loss to the generator training objective.
- Logged the feature-matching term as `g_fm_loss` in training metrics.
- Updated generation so `src/models/generate.py` prefers `checkpoints/model_final.pt`.
- Kept the stable training setup with cuDNN RNNs disabled.

## Result Obtained

- Training completed successfully and produced `checkpoints/model_final.pt`.
- The new run was more stable during optimization.
- Synthetic quality did not improve over the earlier baseline.
- Evaluation on the final synthetic samples gave weak results:
  - KS distance: `0.9124`
  - JS divergence: `0.4565`
  - TSTR weighted F1: `0.0354`

## Conclusion

The experiment helped make training cleaner and more stable, but it did not improve the fidelity of the generated data. The earlier baseline remains the better result to report.