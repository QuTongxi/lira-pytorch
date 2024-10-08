# python3 trainme.py --epochs 100
python3 inferenceme.py --n_queries 3
python3 scoreme.py
python3 inference.py --savedir ./exp/cifar10 --n_queries 3
python3 score.py --savedir ./exp/cifar10

python3 plotme.py --quantize 0
python3 plotme.py --quantize 1