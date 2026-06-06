# FAF-CD

Clean release workspace for **FAF-CD: Frequency-Aware Fusion for Change Detection under Imperfect Multimodal Remote Sensing**, accepted to MONTI 2026.

Paper: [arXiv:2606.03114v1](https://arxiv.org/abs/2606.03114v1)

This repository contains the FAF-CD model, training/evaluation entrypoints, final paper configs, the local DINOv3 runtime, and the VMamba/selective-scan sources needed by the released model. It intentionally excludes datasets, logs, paper build artifacts, generated analysis outputs, and unrelated experiments. Checkpoint files are ignored by git; distribute released weights separately, for example as GitHub Release assets.

## Contents

- `train.py`, `train_runner.py`: training entrypoint.
- `eval.py`: evaluation entrypoint for LEVIR-CD, WHU-CD, and BRIGHT; includes optional BRIGHT submission export.
- `configs/faf_cd/`: final paper configs for LEVIR-CD, WHU-CD, and BRIGHT.
- `models/`: FAF-CD DINOv3 encoder wrapper, frequency-aware fusion modules, Mamba decoder, and selective-scan extension source.
- `dataloader/`, `engine/`, `utils/`: runtime support used by the released configs.
- `dinov3/`: minimal local DINOv3 code used to instantiate ConvNeXt-L/Base backbones and deformable-attention ops.

## Environment Setup

The following setup instructions were tested on Ubuntu 24.04.

The clean FAF-CD release includes the needed DINOv3 and selective-scan source files directly, so no git submodule initialization is required.

Create and activate the conda environment:

```bash
conda create -n faf-cd python=3.12
conda activate faf-cd
```

Install PyTorch, CUDA toolkit, and Python dependencies:

```bash
pip install torch torchvision
conda install nvidia::cuda-toolkit=12.8
pip install -r requirements.txt
```

Set CUDA-related environment variables inside the conda environment:

```bash
conda env config vars set CUDA_HOME="$CONDA_PREFIX"
conda env config vars set LD_LIBRARY_PATH="$CONDA_PREFIX/lib64"
conda deactivate
conda activate faf-cd
```

Build the CUDA extensions required by FAF-CD:

```bash
export TORCH_CUDA_ARCH_LIST="12.0"

cd models/encoders/selective_scan
pip install . --no-build-isolation
cd ../../..

pushd dinov3/dinov3/eval/segmentation/models/utils/ops
pip install . --no-build-isolation
popd
```

For `TORCH_CUDA_ARCH_LIST`, use `"12.0"` for RTX 5090 and `"8.6"` for A6000. If you use system CUDA instead of the conda CUDA toolkit, make sure `CUDA_HOME` points to a CUDA installation that includes `nvcc`.

## Data And Weights

Set dataset roots with `.env` or environment variables:

```bash
cp .env.example .env
# edit DATASETS_ROOT if your datasets are elsewhere
```

Expected dataset folders under `DATASETS_ROOT`:

- `LEVIR-CD256`
- `WHU-CD-256`
- `BRIGHT-1024`

DINOv3 pretrained weights are not included in git. Place them at:

- `pretrained/DINOv3/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth` for LEVIR-CD and WHU-CD.
- `pretrained/DINOv3/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth` for BRIGHT.

FAF-CD paper checkpoints are also ignored by git. Download the released checkpoints from [Google Drive](https://drive.google.com/drive/folders/1a7FB7tO9eqbmDK5K0f8bUbrzHDNMW4R6?usp=sharing), put them under `checkpoints/`, and pass them to evaluation with `--checkpoint_path`.

Currently verified checkpoint artifacts:

- [`checkpoints/FAF-CD_LEVIR_best_test.pth`](https://drive.google.com/file/d/1yRGrjX8Zpgs2tLPLpq0pfM0kJ63OVrHw/view?usp=sharing): LEVIR-CD test checkpoint.
- [`checkpoints/FAF-CD_WHU_best_test.pth`](https://drive.google.com/file/d/1Cbsm6KNrN-shq3uOwgPak1Xsxg3BN-1a/view?usp=sharing): WHU-CD test checkpoint.
- [`checkpoints/FAF-CD_BRIGHT_best_val.pth`](https://drive.google.com/file/d/1KoPNV5XIZ0RbV8tnbpUikqLcgEx3Nqqr/view?usp=sharing): BRIGHT validation checkpoint.

## Train

```bash
python train.py -n faf_cd.levir_dinov3_convnext_large
python train.py -n faf_cd.whu_dinov3_convnext_large
python train.py -n faf_cd.bright_dinov3_convnext_base
```

Training logs default to TensorBoard/local files. W&B is optional: install `wandb` and set `C.log_backend = 'wandb'` plus your own entity/project in a private config if needed.

## Evaluate

```bash
python eval.py -n faf_cd.levir_dinov3_convnext_large --split test --checkpoint_path checkpoints/FAF-CD_LEVIR_best_test.pth --legacy_eval_compat
python eval.py -n faf_cd.whu_dinov3_convnext_large --split test --checkpoint_path checkpoints/FAF-CD_WHU_best_test.pth --legacy_eval_compat
python eval.py -n faf_cd.bright_dinov3_convnext_base --split val --checkpoint_path checkpoints/FAF-CD_BRIGHT_best_val.pth
```

Use `--legacy_eval_compat` for the released LEVIR-CD and WHU-CD paper checkpoints. It preserves the input/color and score aggregation semantics used by the original training/evaluation runs.

## Verified Results

Using the current release code and the checkpoints above:

| Dataset | Split | Change IoU | mIoU / Target mIoU | F1 | mAP | Pixel Acc. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LEVIR-CD | test | 85.882 | 92.539 | 92.405 | 89.509 | 99.233 |
| WHU-CD | test | 91.383 | 95.506 | 95.497 | 94.078 | 99.642 |
| BRIGHT | val | - | 58.464 | - | 76.906 | 96.652 |

Per-target-class IoU for BRIGHT: intact `80.049`, damaged `35.240`, destroyed `60.103`.

## Acknowledgements

FAF-CD builds on our previous work [NeXt2Former-CD](https://github.com/Leeffkkk/NeXt2Former-CD) and is adapted from [M-CD](https://github.com/JayParanjape/M-CD/), which in turn builds on [Sigma](https://github.com/zifuwan/Sigma). We thank the authors of these projects for their valuable contributions and for open-sourcing their implementations. The dataset links referenced above are sourced from [DDPM-CD](https://github.com/wgcban/ddpm-cd), and we thank the authors for making the processed splits easily accessible.

This material is based upon work supported by the National Science Foundation under NSF EIR Grant No. 2401835, entitled "Mapping of Natural Disasters by Deep Subspace Learning in Multi-band and Multi-spectral Satellite Images."

## Citation

```bibtex
@misc{wang2026fafcdfrequencyawarefusionchange,
      title={FAF-CD: Frequency-Aware Fusion for Change Detection under Imperfect Multimodal Remote Sensing},
      author={Yufan Wang and Sokratis Makrogiannis and Chandra Kambhamettu},
      year={2026},
      eprint={2606.03114},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.03114},
}
```
