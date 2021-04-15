Official implementation of [Adversarial Generation of Continuous Images](https://arxiv.org/abs/2011.12026)

### Disclaimer
Right now, the repo is being rewritten/refactored from `firelab` to `pytorch-lightning`, so you may like to wait a week or two for the update before starting building it.

### INR-based GAN

Generating images in their implicit form.


To run the reconstruction model, you should first install `firelab` library:
```
pip install firelab
```

To run the experiment on a single GPU, use the following command:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 --mpi-args="--oversubscribe" python src/run.py -c configs/inr-gan.yml --config.dataset lsun_church_outdoor --config.distributed_training.enabled true --config.logging.freqs.images 1000 --config.logging.freqs.fid 10000
```

To run a multi-gpu training, we use [horovod](https://github.com/horovod/horovod) which is launched via:
```
horovodrun -np NUM_GPUS --mpi-args=--oversubscribe python src/run.py -c configs/inr-gan.yml --config.distributed_training.enabled true --config.dataset lsun_256 --config.hp.batch_size BATCH_SIZE
```
