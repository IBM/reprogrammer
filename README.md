## Model Reprogramming Outperforms Fine-tuning on Out-of-distribution Data in Text-Image Encoders
Andrew Geng, Pin-Yu Chen

![Reprogrammer Methodology](/figures/reprogrammer.png)

This repo contains the reference source code in PyTorch for [The Hidden Costs on Distributional Shifts when Fine-tuning Joint Text-Image Encoders and Redemptions]() (ECCV AROW 22) and [Model Reprogramming Outperforms Fine-tuning on Out-of-distribution Data in Text-Image Encoders]() (SatML 24). 

### Dependencies

The code is built with the following libraries:

- [python=3.8.12](https://www.python.org/)
- [torch==1.11.0](https://pytorch.org/)
- [torchvision=0.17](https://pytorch.org/vision/stable/index.html)
- [ftfy=6.1.3](https://pypi.org/project/ftfy/)
- [regex=2022.7.9](https://pypi.org/project/regex/)
- [tqdm=4.66.2](https://pypi.org/project/tqdm/)

### Usage

##### Get Started

- To train a base reprogrammer model, run the following

```
python reprogramming.py --name=reprogrammer --in-dataset=ImageNet --image-resolution=128 --up-resolution=224 --mr-resolution=192 >> ./reprogrammer.out
```

- To evaluate OOD Generalization for reprogrammer and residual reprogrammer, run the following

```
python evaluate_robustness.py --name=reprogrammer --in-dataset=ImageNet --ood-dataset=ImageNetV2 --image-resolution=128 --method=rp >> robustness_rp.out
python evaluate_robustness.py --name=reprogrammer --in-dataset=ImageNet --ood-dataset=ImageNetV2 --image-resolution=128 --method=resrp >> robustness_rrp.out
```
- To evaluate OOD Detection for reprogrammer and residual reprogrammer, run the following

```
python evaluate_ood.py --name=reprogrammer --in-dataset=ImageNet --image-resolution=128 --method=rp >> ./detection_rp.out
python evaluate_ood.py --name=reprogrammer --in-dataset=ImageNet --image-resolution=128 --method=resrp >> ./detection_rrp.out
```
- To display OOD Detection metrics for reprogrammer and residual reprogrammer, run the following
```
python compute_metrics.py --name=reprogrammer --in-dataset=ImageNet --method=rp
python evaluate_ood.py --name=reprogrammer --in-dataset=ImageNet --method=resrp
```

### Citing

If you find our codebase useful, please consider citing our work:

```
@inproceedings{
    geng2024rp,
    title={Model Reprogramming Outperforms Fine-tuning on Out-of-distribution Data in Text-Image Encoders},
    author={Andrew Geng and Pin-Yu Chen},
    booktitle={SatML},
    year={2024}
}
```
