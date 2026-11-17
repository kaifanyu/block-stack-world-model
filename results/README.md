# Release evidence

`candidates.json` inventories the completed development evaluations used for
this release. Each selected policy folder contains its original protocol and
summary (`selection.json`) and model/training configuration (`training.json`).
Full episode results, original run records and training logs are alongside the
weights on Hugging Face. Original paths are preserved as provenance and are not
runtime dependencies. Incomplete runs and the failed shortcut student are not
released as best checkpoints.

The final integration checks load all seven files, execute one 20-control
training-scene smoke episode per action policy, check model training gradients
and save/reload parity, and run the test suite. They do not replace the historical
200-episode development measurements or provide a new final-test score.
