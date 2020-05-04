python main_normal.py --dataset cifar10 --model resnet --save cifar10_resnet44_bs2048_baseline --epochs 100 --b 2048 --no-lr_bb_fix;
python main_normal.py --dataset cifar10 --model resnet --save cifar10_resnet44_bs2048_lr_fix --epochs 100 --b 2048 --lr_bb_fix;
python main_normal.py --dataset cifar10 --model resnet --save cifar10_resnet44_bs2048_regime_adaptation --epochs 100 --b 2048 --lr_bb_fix --regime_bb_fix;
python main_gbn.py --dataset cifar10 --model resnet --save cifar10_resnet44_bs2048_ghost_bn256 --epochs 100 --b 2048 --lr_bb_fix --mini-batch-size 256;
python main_normal.py --dataset cifar100 --model resnet --save cifar100_wresnet16_4_bs1024_regime_adaptation --epochs 100 --b 1024 --lr_bb_fix --regime_bb_fix;
python main_normal.py --model mnist_f1 --dataset mnist --save mnist_baseline_bs2048_no_lr_fix --epochs 50 --b 2048 --no-lr_bb_fix;
python main_normal.py --model mnist_f1 --dataset mnist --save mnist_baseline_bs2048 --epochs 50 --b 2048 --lr_bb_fix;
python main_gbn.py --model mnist_f1 --dataset mnist --save mnist_baseline_bs4096_gbn --epochs 50 --b 4096 --lr_bb_fix --no-regime_bb_fix --mini-batch-size 128;
python main_gbn.py --model cifar100_shallow --dataset cifar100 --save shallow_cifar100_baseline_bs4096_gbn --epochs 200 --b 4096 --lr_bb_fix --no-regime_bb_fix --mini-batch-size 128;
python main_gbn.py --model cifar10_shallow --dataset cifar10 --save shallow_cifar10_baseline_bs4096_gbn --epochs 200 --b 4096 --lr_bb_fix --no-regime_bb_fix --mini-batch-size 128;
