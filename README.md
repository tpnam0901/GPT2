
<h1 align="center">
  PrivateGPT
  <br>
</h1>

<h4 align="center">New student assignment - GPT2 from scratch</a> </h4>

<p align="center">
<a href=""><img src="https://img.shields.io/github/stars/namphuongtran9196/privategpt?" alt="stars"></a>
<a href=""><img src="https://img.shields.io/github/forks/namphuongtran9196/privategpt?" alt="forks"></a>
<a href=""><img src="https://img.shields.io/github/license/namphuongtran9196/privategpt?" alt="license"></a>
</p>

<p align="center">
  <a href="#how-to-use">How To Use</a> •
  <a href="#references">References</a> •
</p>

## How To Use
- Requirements
```
torch >=2.2.2
CUDA >= 11.5
python >= 3.10
```
- Clone this repository 
```bash
git clone https://github.com/tpnam0901/PrivateGPT.git
cd PrivateGPT
```
- Create a conda environment and install requirements
```bash
conda env create --name py3.10 --file=environments.yml
conda activate py3.10
```

- Export pretrained model
```bash
# Please change the gpt_type in line 12 to match your gpt version
cd src/tools python export_weight.py
```

- Download and preprocess dataset
```bash
mkdir -p src/working/dataset/openwebtext
cd src/tools && python prepare_openwebtext.py --out_dir "../working/dataset/openwebtext"
```

- Before starting training, you need to modify the [config file](./src/configs/base.py) in the config folder. You can refer to the config file in the config folder for more details.

```bash
cd src && python train.py -cfg ../src/configs/4m-ser_bert_vggish.py
```

- Single GPU training
```bash
CUDA_VISIBLE_DEVICES=7 python train.py -cfg configs/openwebtext_gpt2.py

- Multi GPU training
```bash
cd src
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 train.py -cfg configs/openwebtext_gpt2.py
```

- Inference
```bash
CUDA_VISIBLE_DEVICES=4 python infer.py "Tell me a funny story" -cfg_path working/checkpoints/GPT2-openwebtext_gpt2/20240828-044348/cfg.log --best_ckpt --max_new_tokens 50
```

## References

[1] NanoGPT - A simplest, fastest repository for training/finetuning medium-sized GPTs. Available https://github.com/karpathy/nanoGPT.git

---

> GitHub [@tpnam0901](https://github.com/tpnam0901) &nbsp;&middot;&nbsp;
