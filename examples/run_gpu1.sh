# CUDA_VISIBLE_DEVICES=1 python examples/voting_cifar10_cnn.py --optimizer gtk --seed 0 --scale 10 --fullrank
# CUDA_VISIBLE_DEVICES=1 python examples/voting_cifar10_cnn.py --optimizer gtk --seed 0 --scale 10 --fullrank --fixedRV
# CUDA_VISIBLE_DEVICES=1 python examples/voting_cifar10_cnn.py --optimizer gtk --seed 0 --scale 10 --fullrank --weightonly
# CUDA_VISIBLE_DEVICES=1 python examples/voting_cifar10_cnn.py --optimizer gtk --seed 0 --scale 10 --fullrank --fixedRV --weightonly
# CUDA_VISIBLE_DEVICES=1 python examples/voting_cifar10_cnn.py --optimizer sgd --seed 0


CUDA_VISIBLE_DEVICES=1 python examples/voting_cifar10_cnn.py --optimizer gtk --seed 0 --lr 0.01 --fixedRV --diag --exp 1.0
CUDA_VISIBLE_DEVICES=1 python examples/voting_cifar10_cnn.py --optimizer gtk --seed 0 --lr 0.005 --fixedRV --diag --exp 1.0
CUDA_VISIBLE_DEVICES=1 python examples/voting_cifar10_cnn.py --optimizer gtk --seed 0 --lr 0.01 --fixedRV --weightonly --diag --exp 1.0
CUDA_VISIBLE_DEVICES=1 python examples/voting_cifar10_cnn.py --optimizer gtk --seed 0 --lr 0.005 --fixedRV --weightonly --diag --exp 1.0
