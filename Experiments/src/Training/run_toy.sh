#nohup python run_GMM.py -n 32768 -d 8 -s 1 -de 128 -O Adam -B 512 -t -1 --model_order 2 > gmm4_train.log 2>&1 &
#nohup python run_GMM.py -n 16384 -d 8 -s 1 -de 128 -O Adam -B 512 -t -1  --model_order 2 > gmm3_train.log 2>&1 &
#nohup python run_GMM.py -n 8192 -d 8 -s 1 -de 128 -O Adam -B 512 -t -1  --model_order 2 > gmm2_train.log 2>&1 &
nohup python run_GMM.py -n 4096 -d 8 -s 1 -de 128 -O Adam -B 512 -t -1  --model_order 2 > gmm1_train.log 2>&1 &
