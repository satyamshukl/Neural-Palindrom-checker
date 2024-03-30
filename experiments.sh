# # !!!!!! SELECT ALL AND UNCOMMENT !!!!!!!

# # Oversampling
# python train.py --name oversample0 --learning_rate 0.1 --momentum 0.9 --epochs 1000 --batch_size 64 --num_folds 4 --threshold 0.4 --oversample False
# python train.py --name oversample1 --learning_rate 0.1 --momentum 0.9 --epochs 1000 --batch_size 64 --num_folds 4 --threshold 0.4 --oversample True

# python utilities/generate_plot.py --name oversample --num_exps 2 --labels  raw oversampled

# # Batch Size
# python train.py --name bs0 --learning_rate 0.1 --momentum 0.9 --epochs 1000 --batch_size 32 --num_folds 4 --threshold 0.4 --oversample False
# python train.py --name bs1 --learning_rate 0.1 --momentum 0.9 --epochs 1000 --batch_size 64 --num_folds 4 --threshold 0.4 --oversample False
# python train.py --name bs2 --learning_rate 0.1 --momentum 0.9 --epochs 1000 --batch_size 256 --num_folds 4 --threshold 0.4 --oversample False
# python train.py --name bs3 --learning_rate 0.1 --momentum 0.9 --epochs 1000 --batch_size 1024 --num_folds 4 --threshold 0.4 --oversample False

# python utilities/generate_plot.py --name bs --num_exps 4 --labels  32 64 256 1024

# # Epochs
# python train.py --name epoch4 --learning_rate 0.1 --momentum 0.9 --epochs 3000 --batch_size 32 --num_folds 4 --threshold 0.4 --oversample False

# python utilities/generate_plot.py --name epoch --num_exps 1 --labels  3000

# python utilities/generate_plot.py --name epoch --num_exps 8 --labels ['']

# # Learning Rate
# python train.py --name lr0 --learning_rate 0.001 --momentum 0.9 --epochs 1000 --batch_size 32 --num_folds 4 --threshold 0.4 --oversample False
# python train.py --name lr1 --learning_rate 0.01 --momentum 0.9 --epochs 1000 --batch_size 32 --num_folds 4 --threshold 0.4 --oversample False
# python train.py --name lr2 --learning_rate 0.1 --momentum 0.9 --epochs 1000 --batch_size 32 --num_folds 4 --threshold 0.4 --oversample False
# python train.py --name lr3 --learning_rate 0.5 --momentum 0.9 --epochs 1000 --batch_size 32 --num_folds 4 --threshold 0.4 --oversample False

# python utilities/generate_plot.py --name lr --num_exps 4 --labels  0.001 0.01 0.1 0.5


# # Momentum 
# python train.py --name momentum0 --learning_rate 0.1 --momentum 0.00 --epochs 1000 --batch_size 32 --num_folds 4 --threshold 0.4 --oversample False
# python train.py --name momentum1 --learning_rate 0.1 --momentum 0.80 --epochs 1000 --batch_size 32 --num_folds 4 --threshold 0.4 --oversample False
# python train.py --name momentum2 --learning_rate 0.1 --momentum 0.85 --epochs 1000 --batch_size 32 --num_folds 4 --threshold 0.4 --oversample False
# python train.py --name momentum3 --learning_rate 0.1 --momentum 0.90 --epochs 1000 --batch_size 32 --num_folds 4 --threshold 0.4 --oversample False
# python train.py --name momentum4 --learning_rate 0.1 --momentum 0.95 --epochs 1000 --batch_size 32 --num_folds 4 --threshold 0.4 --oversample False
# python train.py --name momentum5 --learning_rate 0.1 --momentum 0.99 --epochs 1000 --batch_size 32 --num_folds 4 --threshold 0.4 --oversample False
# python utilities/generate_plot.py --name momentum --num_exps 6 --labels  0.00 0.80 0.85 0.90 0.95 0.99
