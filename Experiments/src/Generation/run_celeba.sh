nohup python generate.py -D CelebA -n 2048 -i 0 -s 32 -B 512 -LR 0.0001 -O Adam -W 32 -Ns 1 --model_order 2 --device cuda:0 > celeba_gen.log 2>&1 &
