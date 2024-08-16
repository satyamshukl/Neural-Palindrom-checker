# Neural Palindrom Checker

This repository contains scripts and instructions for training a neural network to identify whether a given 10-bit binary string is a palindrome or not. 

**NOTE:** Usage is not limited to a 10-bit binary string. Users can change the architecture to train and identify any length binary string using command-line arguments. In such cases, users have to generate data separately. Refer to the dataset folder for any help.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py --input_size 10 --hidden_size 2 --name exp1 --learning_rate 0.1 --momentum 0.9 --epochs 1000 --batch_size 32 --num_folds 4 --threshold 0.4 --oversample False
```

These are the hyperparameter values we used. Feel free to change the values and experiment with them.

## Hyperparameter Tuning Experiments

To reproduce the results which helped us determine the best hyperparameters, run the following script:

```bash
sh experiments.sh
```

**NOTE:** This will take some time to complete. Please go through the `experiments.sh` file to have a better idea.

Then go to the `hyperparameter_tuning` folder to view the results.
