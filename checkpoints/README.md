# FAF-CD checkpoints

Checkpoint files are ignored by git. Download the released checkpoints from [Google Drive](https://drive.google.com/drive/folders/1a7FB7tO9eqbmDK5K0f8bUbrzHDNMW4R6?usp=sharing), put them here, or pass any checkpoint path directly with `eval.py --checkpoint_path`.

Current local release assets:

- [`FAF-CD_LEVIR_best_test.pth`](https://drive.google.com/file/d/1yRGrjX8Zpgs2tLPLpq0pfM0kJ63OVrHw/view?usp=sharing): verified on LEVIR-CD test with change IoU `85.882`, F1 `92.405`, and pixel accuracy `99.233`.
- [`FAF-CD_WHU_best_test.pth`](https://drive.google.com/file/d/1Cbsm6KNrN-shq3uOwgPak1Xsxg3BN-1a/view?usp=sharing): verified on WHU-CD test with change IoU `91.383`, F1 `95.497`, and pixel accuracy `99.642`.
- [`FAF-CD_BRIGHT_best_val.pth`](https://drive.google.com/file/d/1KoPNV5XIZ0RbV8tnbpUikqLcgEx3Nqqr/view?usp=sharing): verified on BRIGHT validation with target mIoU `58.464`, mAP `76.906`, and pixel accuracy `96.652`.

Use `--legacy_eval_compat` when evaluating the released LEVIR-CD and WHU-CD checkpoints.
