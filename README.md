## Adversarial Generation of Continuous Images [CVPR 2021]

This repo contains [INR-GAN](https://arxiv.org/abs/2011.12026) implementation built on top of [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) repo.

<div style="text-align:center">
<img src="assets/inr-gan.jpg" alt="INR-GAN illustration" width="500"/>
</div>

### Performance
FID scores for the model (with the default nearest neighbour interpolation, i.e. all pixels are generated completely independently) compared to other baselines are presented in the table below.

| Model       | LSUN Churches 256x256 | FFHQ 256x256 | #imgs/sec on V100 32gb | Memory usage             |
| ----------- | --------------------- | ------------ | ---------------------- | ------------------------ |
| INR-GAN     | 4.45                  | 12.46        | 301.69 @ batch_size=50 | 23.54 Gb @ batch_size=50 |
| StyleGAN2   | 3.86                  | 3.83         | 95.79 @ batch_size=32  | 3.65 Gb @ batch_size=32  |
| CIPS        | 2.92                  | 4.38         | 27.27 @ batch_size=16  | 8.11 Gb @ batch_size=16  |

The inference speed in terms of #imgs/sec was measured on a single NVidia V100 GPU (32 Gb) *without* using the mixed precision (see the #Profiling section below).
The FID performance can be improved by using bilinear interpolation, but this deviates from the INR "paradigm" and makes the generation slower.
Note: the above CIPS implementation is not exact. See [CIPS](https://github.com/saic-mdal/CIPS) for the exact one.

For INR-GAN, memory usage is increased for 2 reasons:
- we use coordinate embeddings for high-resolutions
- we cache coordinate embeddings at test time (when they do not depend on z)

Note that the profiling results can differ depending on the hardware and drivers installed (we used CUDA 10.1.243).

### Installation
To install, run the following command:
```
conda env create --file environment.yaml --prefix ./env
conda activate ./env
```

### Training
To train the model, navigate to the project directory and run:
```
python src/infra/launch_local.py hydra.run.dir=. +experiment_name=my_experiment_name +dataset.name=dataset_name num_gpus=4
```
where `dataset_name` is the name of the dataset without `.zip` extension inside `data/` directory (you can easily override the paths in `configs/main.yml`).
So make sure that `data/dataset_name.zip` exists and should be a plain directory of images.
See [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) repo for additional data format details.
This training command will create an experiment inside `experiments/` directory and will copy the project files into it.
This is needed to isolate the code which produces the model.

### Pretrained checkpoints
We provide checkpoints for the following datasets:
- [LSUN Churches 256x256](https://vision-cair.s3.amazonaws.com/inr-gan/checkpoints/churches.pkl) with FID = 4.45.
- [LSUN Bedrooms 256x256](https://vision-cair.s3.amazonaws.com/inr-gan/checkpoints/bedrooms.pkl) with FID = 5.71 (setting truncation to 0.9 is crucial for it). For this dataset, we used an additional convolution on top of 128x128 and 256x256 layers and that's why its throughput dropped from 301.69 to 266.45. We also trained this checkpoint with a "half-sized" discriminator (fmaps=0.5), so the quality for it can be improved by training with a larger discriminator.
- [FFHQ 256x256](https://vision-cair.s3.amazonaws.com/inr-gan/checkpoints/ffhq.pkl) with FID = 12.46 (setting truncation to 0.9 improved FID by 0.3). The quality for it can be improved (at the expense of speed) by using additional convolutions for higher-resolution blocks as for LSUN Bedrooms.

We believe that the reason why it works better on Churches compared to other datasets is that this dataset contains more high-frequency details.

### Data format
We use the same data format as the original [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) repo: it is a zip of images.
It is assumed that all data is located in a single directory, specified in `configs/main.yml`.

For completeness, we also provide downloadable links to the datasets:
- [LSUN Churches 256x256](https://vision-cair.s3.amazonaws.com/inr-gan/datasets/church_outdoor_train_256.zip) of size 1.8 GiB. The original source is [https://www.yf.io/p/lsun](https://www.yf.io/p/lsun).
- [LSUN Bedroom 256x256](https://vision-cair.s3.amazonaws.com/inr-gan/datasets/bedroom_train_256.zip) of size 32.8 GiB. The original source is [https://www.yf.io/p/lsun](https://www.yf.io/p/lsun).
- [FFHQ 256x256](https://vision-cair.s3.amazonaws.com/inr-gan/datasets/ffhq_256.zip) of size 6.5 GiB. The original source is [https://github.com/NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset).

Download the datasets and put them into `data/` directory.

### Profiling
To profile the model, run:
```
CUDA_VISIBLE_DEVICES=0 python src/scripts/profile.py hydra.run.dir=. model=inr-gan.yml
```

The inference speed in terms of #imgs/sec was measured on a single NVidia V100 GPU (32 Gb).
Note, that this model was developed *before* StyleGAN2-ADA, i.e. before mixed precision was a thing.
With mixed precision enabled, StyleGAN2 produced 256.88 #imgs/sec @ batch_size=128.
INR-GAN (default architecture) with mixed precision gives only 465.60 #imgs/sec @ batch_size=100 (only 50% speed increase compared to its full-precision version) and we didn't try training it (performance might drop).
We also compared to [CIPS](https://github.com/saic-mdal/CIPS) (which is a parallel work that explores INR-based generation) in terms of speed (didn't try training it).
For all the models, we used the optimal batch size unique for them.

### License
This repo is built on top of [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch), so I assume it is restricted by the [NVidia license](https://nvlabs.github.io/stylegan2-ada-pytorch/license.html) (though I am not a lawyer).


### Bibtex
```
@misc{inr_gan,
      title={Adversarial Generation of Continuous Images},
      author={Ivan Skorokhodov and Savva Ignatyev and Mohamed Elhoseiny},
      year={2020},
      eprint={2011.12026},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{cips,
  title={Image Generators with Conditionally-Independent Pixel Synthesis},
  author={Anokhin, Ivan and Demochkin, Kirill and Khakhulin, Taras and Sterkin, Gleb and Lempitsky, Victor and Korzhenkov, Denis},
  journal={arXiv preprint arXiv:2011.13775},
  year={2020}
}
```
