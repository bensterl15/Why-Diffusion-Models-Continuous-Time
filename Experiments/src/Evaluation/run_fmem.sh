nohup python compute_fmem.py -D CelebA -n 2048 -i 0 -s 32 -LR 0.0001 -O Adam -W 32 -B 512 -Ns 1 --gap_threshold 0.333 --device cuda:0 > celeba_eval.log 2>&1 &
