# HIM

This is the implementation of HIM (TKDE, 2026).

## 1. Environment

Please check `requirements.txt` for the Python packages used in the experiment.

One can create a new Conda environment to run the code as:

```cmd
conda create -n <ENV_NAME> python=3.9 -y
conda activate <ENV_NAME>	
# one should first install PyTorch with version torch==1.13.1+cu117
pip install -r requirements.txt
```

Please replace `<ENV_NAME>` with any custom environment name.

## 2. Run

1. Please extract `configs.zip` and `data.zip` directly and ensure the `data` folder is in the project root directory as:

```
configs/
data/
methods/
...
```

2. To run dataset YouTube, one should first run the script `process_youtube.py` in path `data/youtube/process_youtube.py` to generate the social network for YouTube.

#### Run HIM

1. After activating the environment, run `run.py` to evaluate the HIM. Select the target dataset in `run.py` as:

```python
...
datasets = {
    1: 'cora_ml',
    2: 'power_grid',
    3: 'facebook',
    4: 'github',
    5: 'youtube'
}
data_name = datasets[1]  # to run on dataset Cora-ML
...
```

2. Based on the above selection, an example running command to run HIM on Cora-ML under the WIC model is:

```cmd
python run.py --diffusion_model WIC --seed_ratio 10 --method MD # run HIM_MD
python run.py --diffusion_model WIC --seed_ratio 10 --method ASW # run HIM
```

3. Similarly, one can run HIM under the WLT model as follows:

```cmd
python run.py --diffusion_model WLT --seed_ratio 10 --method MD # run HIM_MD
python run.py --diffusion_model WLT --seed_ratio 10 --method ASW # run HIM
```

#### P.S.

All configurations for each method can be found in `.yaml` files in `configs`.

## 3. Reference

If you find our code useful for your research, please consider citing our paper:

```
@article{qiao_2026_him,
  title   = {Influence Strength Estimation in Hyperbolic Space for Social Influence Maximization},
  author  = {Hongliang Qiao and Shanshan Feng and Min Zhou and Xutao Li and Yunming Ye and Fan Li and Shuo Shang and Yew-Soon Ong},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  year    = {2026},
  doi     = {10.1109/TKDE.2026.3683562}
}
```

