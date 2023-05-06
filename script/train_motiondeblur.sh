#!/bin/bash

pip install timm -i https://pypi.mirrors.ustc.edu.cn/simple/
pip install einops -i https://pypi.mirrors.ustc.edu.cn/simple/
pip install opencv-python -i https://pypi.mirrors.ustc.edu.cn/simple/

cd /code/ShuffleFormer/train


python train_motiondeblur_h.py --arch ShuffleFormer_B_Half --batch_size 8 --gpu '0,1' --train_ps 256 --train_dir /data/JIEXIAO/GoPro/train --val_ps 256 --val_dir /data/JIEXIAO/GoPro/val --env _0706 --mode deblur --nepoch 600 --checkpoint 200 --dataset GoPro --warmup_epochs 150