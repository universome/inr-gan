# DINR

## Dependencies

```bash
python -m venv ~/.venv/dinr-env
source ~/.venv/dinr-env/bin/activate
pip install -r requirements.txt
python setup.py build develop
```

Install PyTorch3d
```bash
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
git checkout b24d89a28319bc9eabe8d64cee6848b682b0903c
pip install -e .
```

## Data
```bash
mkdir data
ln -s {{lsun_path}} data/lsun
```

## List of experiments launched

```bash
python train_net.py trainer.gpus=8 hydra.run.dir=experiments_v2/sg2_original_churches trainer.val_check_interval=15000 model=stylegan2_default system=stylegan2

python train_net.py trainer.gpus=8 hydra.run.dir=experiments_v2/cips_original_churches trainer.val_check_interval=15000 model=stylegan2_cips system=stylegan2_noppl_nomix_inr

python train_net.py trainer.gpus=8 hydra.run.dir=experiments_v2/cips_ne_original_churches trainer.val_check_interval=15000 model=stylegan2_cips_ne system=stylegan2_noppl_nomix_inr
```
