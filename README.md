# Random Shuffle Transformer for Image Restoration (ICML 2023)
 <b><a href='https://jiexiaou.github.io'>Jie Xiao</a>, <a href='https://xueyangfu.github.io'>Xueyang Fu</a>, <a href='https://manman1995.github.io'>Man Zhou</a>, Hongjian Liu, Zheng-Jun Zha</b>
 
## Pretrained Model
- SIDD [Google Drive](https://drive.google.com/file/d/1rK_fwwz70DBoYIR-KqBLB14qC_cXegQD/view?usp=sharing)

## Demo

```
python test_sidd.py --arch ShuffleFormer_B --gpu '0' --val_dir data_path --pretrain_weights model_path --result_dir save_dir --repeat_num 8
```