## CoFi-Lite
[![demo](https://img.shields.io/badge/GitHub-Demo-orange.svg)](https://acceleration123.github.io/CoFiLite-demo/)
[![arxiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2607.10142)

🎉 This is the official implementation of our *IEEE SPL* paper:

[CoFi-Lite: Pushing the Limits of Ultra-Lightweight Speech Enhancement](https://ieeexplore.ieee.org/document/11603399)

## 🔥 News
- [**2026-10-19**] The source codes and checkpoint on the simulated DNS3 dataset are released.
- [**2026-7-14**] The final manuscript is uploaded to arxiv.
- [**2026-7-7**] The manuscipt is accepted by IEEE SPL.

## Introduction

CoFi-Lite is an **ultra-lightweight**, causal speech enhancement model for 16 kHz speech. Two parallel encoder–decoder paths model full-band envelopes and low-frequency details separately, and a Cross-Path Fusion (CPF) module enables efficient feature interaction between them. With only **12.87M MACs/s** and **83.12k parameters**, CoFi-Lite outperforms GTCRN using just 40.26% of its computational complexity.

## Results

Comparison with the ultra-lightweight baseline GTCRN on the simulated DNS3 test set and the DNS Challenge 2020 test set. Best result per metric is in **bold**. Lower is better for MACs/s, Params and RTF; higher is better for the rest.

**Simulated DNS3 test set**

| Model | MACs/s (M) ↓ | Params (k) ↓ | RTF ↓ | PESQ ↑ | ESTOI (%) ↑ | SI-SNR ↑ | DNSMOS P.808 ↑ | DNSMOS P.835 (OVRL/SIG/BAK) ↑ |
|---|---|---|---|---|---|---|---|---|
| Noisy | - | - | - | 1.40 | 66.90 | 5.61 | 2.82 | 1.63 / 2.05 / 1.86 |
| GTCRN | 31.97 | **23.67** | 0.050 | 2.07 | 75.11 | 11.30 | 3.48 | 2.63 / 2.98 / 3.81 |
| CoFi-Lite | **12.87** | 83.12 | **0.033** | **2.16** | **76.10** | **11.80** | **3.53** | **2.70** / **3.05** / **3.85** |

**DNS Challenge 2020 test set (DNSMOS P.835)**

| Model | No Reverb (OVRL/SIG/BAK) ↑ | With Reverb (OVRL/SIG/BAK) ↑ |
|---|---|---|
| Noisy | 2.48 / 3.39 / 2.62 | 1.39 / 1.76 / 1.50 |
| GTCRN | 3.09 / 3.38 / 4.01 | 2.43 / 2.86 / 3.52 |
| CoFi-Lite | **3.15** / **3.43** / **4.03** | **2.48** / **2.89** / **3.57** |

## Notes

- **Band split in the BM module:** The paper (Sec. III-B1) describes a 65 + 64
  split with the boundary at 2 kHz, while the released code and checkpoint use
  64 + 65: with `low_lim=2000`, `round(2000 / 16000 * 512) = 64` linear bands
  are kept unaltered and the remaining 193 bins are merged into 65 ERB bands.
  All reported results (Tables I and III) were obtained with this released
  code split. We verified that retraining with the paper's split (65 + 64)
  yields almost identical performance, i.e., the exact placement of the 2 kHz
  boundary bin has little practical impact. To use the paper's split instead,
  set `low_lim=2031.25` (65 / 512 * 16000), which gives 65 + 64 bands — note
  that the released checkpoint is not compatible with this change.

## Repository Structure
```
├── checkpoint/model_on_dns3.tar    # Pretrained checkpoint (simulated DNS3 dataset)
├── configs/
│   ├── cfg_train.yaml              # Training configuration
│   └── cfg_infer.yaml              # Inference configuration
├── model/cofilite.py               # CoFi-Lite model definition
├── dataloader.py                   # DNS3 dataset & data loader
├── train.py                        # Training script (single-/multi-GPU DDP)
├── infer.py                        # Inference script
├── evaluate.py                     # Evaluation wrapper (DNSMOS / intrusive metrics)
├── evaluation/                     # Metric calculation scripts
├── DNSMOS/DNSMOS/                  # Pretrained DNSMOS ONNX models
├── loss_factory.py                 # Hybrid loss (RI + magnitude + SI-SNR)
├── scheduler.py                    # Linear-warmup cosine-annealing LR scheduler
└── distributed_utils.py            # DDP utilities
```

## Inference

1. **Prepare the test data.** Put the noisy 16 kHz WAV files in `noisy_dir`, and the corresponding clean reference WAV files (identical file names) in `clean_dir`. The clean references are only required for the intrusive metrics.

2. **Edit `configs/cfg_infer.yaml`.** Below is a ready-to-use example with the released checkpoint:

   ```yaml
   test_dataset:
     noisy_dir: <path/to/DNS3/test_noisy>     # folder of noisy 16 kHz wavs
     clean_dir: <path/to/DNS3/test_clean>     # folder of clean reference wavs

   network:
     exp_path: <path/to/output>               # any existing directory
     config: configs/cfg_train.yaml           # must be an existing YAML file
     ckpt_name: model_on_dns3                 # checkpoint name (used in enh_folder)
     checkpoint: checkpoint/model_on_dns3.tar # path to the checkpoint (.tar)
     enh_folder: <path/to/output>/enhanced    # output folder for enhanced wavs
   ```

3. **Run inference:**

   ```bash
   python infer.py -C configs/cfg_infer.yaml -D 0
   ```
   `-D` selects the GPU index (falls back to CPU if CUDA is unavailable). The checkpoint is loaded from the `model` entry of the checkpoint dict.

4. **Outputs** (in `network.enh_folder`):
   - `<uid>_enh.wav` — the enhanced utterances;
   - `inf.scp` / `ref.scp` — utterance-id/path lists consumed by `evaluate.py`.

## Evaluation

`evaluate.py` reads `enh_folder` from `configs/cfg_infer.yaml` and must be run **after** `infer.py` so that `inf.scp`/`ref.scp` exist.

```bash
# Non-intrusive DNSMOS (OVRL, SIG, BAK, P808_MOS); clean references not needed
python evaluate.py --metric dnsmos --config configs/cfg_infer.yaml --device 0

# Intrusive metrics (SDR, SI-SNR, PESQ, ESTOI); requires clean references
python evaluate.py --metric intrusive --config configs/cfg_infer.yaml --device 0
```

- `--metric dnsmos` runs `evaluation/calculate_nonintrusive_dnsmos.py` with the pretrained ONNX models in `DNSMOS/DNSMOS/` (already included in this repo). It scores the enhanced signals only. Requires `espnet` (provides `espnet2.enh.layers.dnsmos`), `onnxruntime` and `onnx2torch`.
- `--metric intrusive` runs `evaluation/calculate_intrusive_se_metrics.py`, which aligns each enhanced utterance with its clean reference via `ref.scp` and computes the intrusive metrics in parallel (`--nj 8` workers). 

**Results** are summarized in `RESULTS.txt`:

- `${enh_folder}/scoring_dnsmos/RESULTS.txt` — mean OVRL / SIG / BAK / P808_MOS;
- `${enh_folder}/scoring_intrusive/RESULTS.txt` — mean SDR / SI-SNR / PESQ / ESTOI.

Per-utterance scores are additionally stored as `*.scp` files in the same directories.

## Training

1. **Prepare the data and edit the data paths.** Open `dataloader.py` and set the training/validation folders:

   ```python
   NOISY_DATABASE_TRAIN = '<path/to/DNS3/train_noisy>'
   NOISY_DATABASE_VALID = '<path/to/DNS3/dev_noisy>'
   ```

   > Note: the clean signal of each utterance is located by replacing `noisy` with `clean` in the file path, so the folders must be named accordingly (e.g. `.../train_noisy` and `.../train_clean`).

2. **Edit `configs/cfg_train.yaml`.** The default recipe reproduces the released model:

   | Entry | Default | Description |
   |---|---|---|
   | `optimizer.lr` | 1e-3 | Adam learning rate. |
   | `scheduler.kwargs` | warmup 25k steps, decay until 250k steps, max lr 1e-3, min lr 1e-6 | Linear-warmup cosine-annealing schedule, updated per step. |
   | `loss` | `lamda_ri=30`, `lamda_mag=70`, `compress_factor=0.3` | Hybrid loss: 30 × compressed RI loss + 70 × compressed magnitude MSE + negative SI-SNR. |
   | `train_dataset` | 10 s clips, 72k utterances in total, 10k sampled per epoch | 16 kHz training segments. |
   | `train_dataloader` | batch size 8 | — |
   | `trainer.epochs` | 200 | Total training epochs. |
   | `trainer.clip_grad_norm_value` | 3.0 | Gradient clipping. |
   | `trainer.exp_path` | — | Root directory of the experiment outputs. |

3. **Run training:**

   ```bash
   # Single GPU
   python train.py -C configs/cfg_train.yaml -D 0

   # Multi-GPU (DDP), e.g. 4 GPUs
   python train.py -C configs/cfg_train.yaml -D 0,1,2,3
   ```

4. **Resume training.** Set `trainer.resume: True` and `trainer.resume_datetime` to the timestamp suffix of the experiment directory (e.g. `2026-05-13-18h19m`); the latest checkpoint in that run's `checkpoints/` folder will be loaded and training continues from the next epoch.

## Citation
If you find this work useful, please cite our paper:

```bibtex
@ARTICLE{11603399,
  author={Yang, Leyan and Wang, Dahan and Rong, Xiaobin and Zhao, Jiadong and Lu, Jing},
  journal={IEEE Signal Processing Letters}, 
  title={CoFi-Lite: Pushing the Limits of Ultra-Lightweight Speech Enhancement}, 
  year={2026},
  volume={33},
  number={},
  pages={2954-2958}
}
```

## 📬 Contact
Leyan Yang: [leyan.yang@smail.nju.edu.cn](mailto:leyan.yang@smail.nju.edu.cn)
