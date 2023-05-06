#!/bin/bash  

pip install timm -i https://pypi.mirrors.ustc.edu.cn/simple/
pip install einops -i https://pypi.mirrors.ustc.edu.cn/simple/
pip install opencv-python -i https://pypi.mirrors.ustc.edu.cn/simple/

cd /code/ShuffleFormer/train

python train_denoise_h.py --arch ShuffleFormer_B_Half --batch_size 32 --gpu '0,1' --train_ps 128 --train_dir /data/JIEXIAO/SIDD_Medium/train --env _0706 --val_dir /data/JIEXIAO/SIDD_Medium/test --save_dir /output --warmup --dataset sidd --display_freq 50