import numpy as np
import pickle  

SEED = 18  # Setting a seed for reproducibility
np.random.seed(SEED)  

import os
import random  
import pandas as pd  
import matplotlib.pyplot as plt  
import argparse  
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from model import Palindrome


def precision_recall(targets, predictions):
    """
    Function to calculate precision and recall given true targets and predicted labels.

    Parameters:
        targets : True target labels.
        predictions : Predicted labels.

    Returns:
        Precision and recall values.
    """
    tp = np.sum(np.logical_and(predictions == 1, targets == 1))
    fp = np.sum(np.logical_and(predictions == 1, targets == 0))
    fn = np.sum(np.logical_and(predictions == 0, targets == 1))
    precision = tp / (tp + fp + 1e-16)
    recall = tp / (tp + fn + 1e-16)
    return precision, recall


def train(args, data, epochs=100, batch_size=32, columns=None):
    """
    Function to train the Palindrome model.

    Parameters:
        args : Command-line arguments.
        data : Input data (pandas).
        epochs : Number of training epochs.
        batch_size : Size of each training batch.
    """
    os.makedirs(f'exp/{args.name}', exist_ok=True)
    FOLDS = args.num_folds
    threshold = args.threshold
    best_fold = 0

    final_weights = {
        1: np.zeros((2, 10)),
        2: np.zeros((1, 2))
    }
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    max_precision = 0.0
    max_recall = 0.0

    for fold, (train_indices, test_indices) in enumerate(skf.split(data.iloc[:, :-1], data.iloc[:, -1])):
        fold_loss = [] 
        fold_acc = []
        fold_test_loss = []
        fold_test_acc = []
        fold_precision = []
        fold_recall = []
        model = Palindrome(learning_rate=args.learning_rate, momentum=args.momentum) 
        # Resuming training from saved weights if specified
        if args.resume:
            model.load_weights('palindrome_weights.pkl')
        train_data, test_data = data.iloc[train_indices], data.iloc[test_indices] 
        if args.oversample == 'True':
            print('Oversampling training data...')
            # Oversampling data if specified
            train_data.columns = columns
            train_data = oversampled(train_data, args.oversampling_samples)
            train_data.columns = data.columns
            print('Training on oversampled data.')
        print(len(train_data), len(test_data))
        for epoch in range(epochs):
            indices = np.arange(len(train_data))
            np.random.shuffle(indices)
            total_loss = 0
            acc = 0
            predictions = []
            target_list = []
            for i in range(0, len(train_data), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch = train_data.iloc[batch_indices]
                inputs = batch.iloc[:, :-1].to_numpy().T
                targets = batch.iloc[:, -1].to_numpy().reshape(1, -1)
                outputs = model.forward(inputs)
                preds = np.where(outputs > threshold, 1, 0)
                predictions.extend(list(preds))
                target_list.extend(list(targets))
                acc += (preds == targets.reshape(preds.shape)).sum().item()
                loss = model.loss(targets, outputs)
                total_loss += loss
                model.backward(inputs, targets, outputs)
            total_loss /= len(train_data) / batch_size
            acc /= len(train_data)

            test_inputs = test_data.iloc[:, :-1].to_numpy().T
            test_targets = test_data.iloc[:, -1].to_numpy().reshape(1, -1)
            test_outputs = model.forward(test_inputs)
            test_loss = model.loss(test_targets, test_outputs)
            test_preds = np.where(test_outputs > threshold, 1, 0)
            test_acc = (test_preds == test_targets.reshape(test_preds.shape)).mean().item()
            precision, recall = precision_recall(test_targets, np.where(test_outputs > threshold, 1, 0))

            fold_loss.append(total_loss)
            fold_acc.append(acc)
            fold_test_loss.append(test_loss)
            fold_test_acc.append(test_acc)
            fold_precision.append(precision)
            fold_recall.append(recall)

            if epoch % 100 == 0:
                print(
                    f"Fold {fold + 1}, Epoch {epoch + 1}/{epochs}\tTrain Loss: {total_loss:.4f}, Train Accuracy: {acc:.4f}, Test Accuracy: {test_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        if precision >= max_precision and recall >= max_recall:
            final_weights[1] = model.weights[1]
            final_weights[2] = model.weights[2]
            max_recall = recall
            max_precision = precision

            logs = pd.DataFrame()
            logs['loss'] = fold_loss
            logs['acc'] = fold_acc
            logs['test_loss'] = fold_test_loss
            logs['test_acc'] = fold_test_acc
            logs['precision'] = fold_precision
            logs['recall'] = fold_recall

        print(f'\n\nFOLD {fold + 1}: Test Dataset Classification Report: \n')
        print(classification_report(test_targets.reshape(-1,), test_preds.reshape(-1,)))
        cm = confusion_matrix(test_targets.reshape(-1,), test_preds.reshape(-1,))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(f'exp/{args.name}/cm_{fold}.jpg') # Saving confusion matrix plot
        print(model.weights)
        print()
        logs.to_csv(f'exp/{args.name}/logs.csv', index=False)

    with open(f'exp/{args.name}/palindrome_weights.pkl', 'wb') as f:
        pickle.dump(final_weights, f)  # Saving final model weights

    print(f'\n\nBest fold: {best_fold+1}, Best weights:')
    print(final_weights)


def oversampling_data_uniformly(data, samples=1024):
    """
    Function to perform uniform oversampling on the input data.

    Parameters:
        data (pandas.DataFrame): Input data.
        samples (int): Number of samples for oversampling.

    Returns:
        Oversampled data as a Pandas DataFrame.
    """
    positive = np.array(data[data['1'] == 1])
    negatives = np.array(data[data['1'] == 0])
    oversample = []
    for i in range(samples):
        random_id_pos = int(random.uniform(0, 1) * len(positive))
        random_id_neg = int(random.uniform(0, 1) * len(negatives))
        oversample.append(positive[random_id_pos])
        oversample.append(negatives[random_id_neg])
    return np.array(oversample)


def oversampled(data, number_of_samples=30):
    """
    Function to perform oversampling on the input data.

    Parameters:
        data (pandas.DataFrame): Input data.
        number_of_samples (int): Number of times to repeat the positive samples.

    Returns:
        Oversampled data as a Pandas DataFrame.
    """
    positive = np.array(data[data['1'] == 1])
    negatives = np.array(data[data['1'] == 0])
    oversample = []
    for i in range(number_of_samples):
        oversample += list(positive)
    oversample += list(negatives)
    return pd.DataFrame(oversample)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Palindrome model.')
    parser.add_argument('--name', type=str, default="test", help='name of the experiment')
    parser.add_argument('--learning_rate', type=float, default=0.5, help='learning rate for training')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for training')
    parser.add_argument('--epochs', type=int, default=2000, help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--num_folds', type=int, default=4, help='number of folds for cross-validation')
    parser.add_argument('--threshold', type=float, default=0.4, help='threshold for predictions')
    parser.add_argument('--oversampling_samples', type=int, default=31, help='number of samples for oversampling')
    parser.add_argument('--resume', type=bool, default=False, help='resume from last saved weights')
    parser.add_argument('--oversample', type=str, default='False', help='resume from ;last saved weights')

    args = parser.parse_args()
    print(args.oversample)

    # model = Palindrome(learning_rate=args.learning_rate, momentum=args.momentum, threshold=args.threshold)
    data = pd.read_csv('palindrome_data.csv')

    train(args, data, epochs=args.epochs, batch_size=args.batch_size, columns=data.columns)
