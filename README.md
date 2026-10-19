## CoFi-Lite
[![demo](https://img.shields.io/badge/GitHub-Demo-orange.svg)](https://acceleration123.github.io/CoFiLite-demo/)
[![arxiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2607.10142)

🎉 This is the official implementation of our *IEEE SPL* paper:

[CoFi-Lite: Pushing the Limits of Ultra-Lightweight Speech Enhancement](https://ieeexplore.ieee.org/document/11603399)

## 🔥 News
- [**2026-10-19**] The source codes and checkpoint on the simulated DNS3 dataset are released.
- [**2026-7-14**] The final manuscript is uploaded to arxiv.
- [**2026-7-7**] The manuscipt is accepted by IEEE SPL.

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
