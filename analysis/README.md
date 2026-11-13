# Policy analysis

- `Privileged FM Policy/`: privilegeFM, formerly L2.
- `Encoder FM Policy/`: encoderFM, formerly L4.
- `rs-imle/`, `flow/`, `imagination/`, `teacher-ppo/`: existing other families.
- simulatorFM/L3 runs are archived at `/workspace/artifacts/L3`, outside this view.

Run `python scripts/setup_analysis.py` to recreate local per-run links from
`catalog.json`. This does not move or copy data. Some linked runs are still
running: inspect them, but do not modify them. Links are machine-local and
ignored by Git. New runs write into new real directories in the two FM folders.
Historical file contents and checkpoint metadata retain their original names.

## Hugging Face

For policies, prefer one model repository per selected run, with its final
checkpoint, configuration/normalization, evaluation summaries, and model card.
Intermediate `step-*.pt` and optimizer `training.pt` files need not be uploaded
for inference. The helper previews a completed run without network access:

```bash
python scripts/upload_policy.py --run "analysis/Encoder FM Policy/<completed-run>"
```

To upload, authenticate using `hf auth login`, then explicitly request upload:

```bash
python scripts/upload_policy.py \
  --run "analysis/Encoder FM Policy/<completed-run>" \
  --repo-id YOUR_ACCOUNT/encoderFM-RUN_NAME --upload
```

The helper stages a separate copy, checks that source files did not change,
adds hashes and a model card, and creates/requires a private model repository.
It refuses unfinished runs. It uses existing Hugging Face authentication and
never embeds credentials. Temporary staging requires space for the selected
files. A frozen/adapter policy also needs the external JEPA encoder identified
by its metadata; the helper does not bundle that checkpoint, simulator assets,
or datasets. These are custom PyTorch policies, not Transformers models.

For an entire archive, first prepare a stable local export, then use Hugging
Face's folder uploader. Do not upload the live symlink view as if it were a
portable snapshot. Large-folder uploads support resumable work; account storage
limits still apply. See the official guide:
https://huggingface.co/docs/huggingface_hub/main/guides/upload

No Hugging Face repository was created or uploaded by this cleanup.
