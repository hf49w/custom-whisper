# custom-whisper

Server-ready multimodal ASR training bundle based on Whisper with frozen encoder/decoder and trainable multimodal fusers.

## Included training suite

The main server entrypoint is `scripts/run_flickr8k_subset_requested_models.sh`. It:

- builds a Flickr8k manifest from local image/audio/caption files
- selects a subset with exactly `2` audio rows per image
- splits the subset into train/test grouped by `image_id`
- trains for exactly `5` epochs
- keeps the Whisper audio encoder, Whisper text decoder, and visual encoder frozen

It runs these five requested model combinations:

- `resnet50 + proj_concat_proj`
- `resnet_gmlp + concat_temp`
- `clip + cross_attn_gate`
- `clip + attn_prefix`
- `clip + gated_seq_concat`

## Repository layout

- `custom_whisper/`: multimodal Whisper package with the added fusion modules
- `scripts/`: data preparation, subset selection, training, evaluation, and server run scripts
- `data/`: empty dataset/model directories for manual transfer
- `outputs/`: training outputs, checkpoints, and summaries
- `envs/create_conda_env.sh`: Linux server conda setup script

## Expected data layout

Put your files into these locations before training:

- `data/flickr8k/images/`: Flickr8k image files
- `data/flickr8k/audio/`: Flickr8k wav files
- `data/flickr8k/captions/captions.txt`: caption CSV with columns `image,caption`

Optional local model caches:

- `data/models/clip/`: if you want to place a local CLIP checkpoint and override `CLIP_MODEL_NAME`
- `data/models/whisper/`: local Whisper download cache

## Server setup

Create the environment on the server:

```bash
bash envs/create_conda_env.sh
conda activate custom-whisper-mm
```

If the server can reach Hugging Face/OpenAI model hosts, the default scripts can download Whisper and CLIP automatically on first run. If you need offline training, pre-populate the caches and run with `OFFLINE=1`.

## Train

Run the full requested suite:

```bash
bash scripts/run_flickr8k_subset_requested_models.sh
```

Common overrides:

```bash
PYTHON_BIN=python \
BATCH_SIZE=2 \
SAVE_EVERY_BATCHES=50 \
OFFLINE=0 \
WHISPER_MODEL=medium.en \
CLIP_MODEL_NAME=openai/clip-vit-base-patch32 \
bash scripts/run_flickr8k_subset_requested_models.sh
```

Outputs are written under `outputs/<suite_tag>/`.

## Resume an interrupted run

`scripts/run_flickr8k_custom_whisper_fuser.sh` will automatically resume from:

```text
<experiment_root>/model/checkpoints/last.pt
```

The training script now refreshes `last.pt` every `SAVE_EVERY_BATCHES` completed batches, so a killed job can resume inside the same epoch instead of restarting that epoch from batch 1.

Manual resume example:

```bash
python scripts/train_visspeech_custom_whisper_fuser.py \
  --train-manifest data/flickr8k/prepared/subsets/2_per_image_random_sel42_split42_test20/train_manifest.jsonl \
  --output-root outputs/my_resume_run/model \
  --whisper-model medium.en \
  --visual-encoder clip \
  --visual-fuser cross_attn_gate \
  --resume-from outputs/my_resume_run/model/checkpoints/last.pt
```
