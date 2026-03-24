nohup python compute_FID.py \
  -D CelebA \
  -n 2048 \
  -i 0 \
  -s 32 \
  -LR 0.0001 \
  -O Adam \
  -W 32 \
  -B 512 \
  -m 2 \
  -istat 1 \
  --device cuda:0 \
  > celeba_FID.log 2>&1 &
