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
