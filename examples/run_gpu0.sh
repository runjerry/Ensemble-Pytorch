# CUDA_VISIBLE_DEVICES=0 python examples/voting_cifar10_cnn.py --optimizer gtk --seed 0 --diag --exp 1.0
# CUDA_VISIBLE_DEVICES=0 python examples/voting_cifar10_cnn.py --optimizer gtk --seed 0 --lr 0.01 --fixedRV --diag --exp 1.0
# CUDA_VISIBLE_DEVICES=0 python examples/voting_cifar10_cnn.py --optimizer gtk --seed 0 --lr 0.005 --fixedRV --diag --exp 1.0
# CUDA_VISIBLE_DEVICES=0 python examples/voting_cifar10_cnn.py --optimizer gtk --seed 0 --weightonly --diag --exp 1.0
# CUDA_VISIBLE_DEVICES=0 python examples/voting_cifar10_cnn.py --optimizer gtk --seed 0 --lr 0.01 --fixedRV --weightonly --diag --exp 1.0
# CUDA_VISIBLE_DEVICES=0 python examples/voting_cifar10_cnn.py --optimizer gtk --seed 0 --lr 0.005 --fixedRV --weightonly --diag --exp 1.0

CUDA_VISIBLE_DEVICES=0 python examples/voting_cifar10_cnn.py --optimizer sgd --seed 0 --lr 0.01
CUDA_VISIBLE_DEVICES=0 python examples/voting_cifar10_cnn.py --optimizer sgd --seed 0 --lr 0.005
CUDA_VISIBLE_DEVICES=0 python examples/voting_cifar10_cnn.py --optimizer sgd --epoch 400 --seed 0

