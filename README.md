# Official repository for ADAPT to Robustify Prompt Tuning Vision Transformers (TMLR 2025) <a>https://openreview.net/forum?id=bZzXgheUSD</a>

This code is for prompt-tuning of vision transformers robustly. The key is to avoid gradient obfuscation by conditioning on the prompt during the attack generation process

-------
To train a vit base model using ADAPT CE you can use the following line 
```
python train.py --model "vit_base_patch16_224_in21k" --method ADAPT --adapt-loss ce --params P2T  --seed 0 --prompt_length 25 --epochs 20  --chkpnt_interval 20 --lr-schedule cyclic --lr-max 1  --train-patch --dataset cifar10
```
To evaluate the model simply add `--just-eval`, `--load`, and specify the path to the checkpoint with `--loadpath` to the same line above.
-------
Cite our work:
```
@article{
eskandar2025adapt,
title={{ADAPT} to Robustify Prompt Tuning Vision Transformers},
author={Masih Eskandar and Tooba Imtiaz and Zifeng Wang and Jennifer Dy},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=bZzXgheUSD},
note={}
}
```
