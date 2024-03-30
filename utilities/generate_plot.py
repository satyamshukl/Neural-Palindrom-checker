import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt

def graph_for_metric(args, metric, metric_name):
    # labels = [label for label in args.label.split(',')]
    plt.clf()
    # plt.figure(figsize=(30,30))
    for k, value in enumerate(metric):
        xs = list(range(len(value)))
        # lw = len(args.labels) / (k+1)
        # ls = ['-','--','-.',':'][k % 4]
        plt.plot(xs, value, label=args.labels[k]) #, linestyle=ls, linewidth=lw)
        
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(metric_name)

    plt.legend()
    path = f'hyperparameter_tuning/{args.name}/'
    # plt.show()
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + f'{metric_name}.jpg')
    # plt.close()
    

def generate_graph(args):
    loss = []
    acc = []
    test_loss = []
    test_acc = []
    precision = []
    recall = []

    for i in range(args.num_exps):
        logs = pd.read_csv(f'exp/{args.name}{i}/logs.csv')
        loss.append(list(logs['loss']))
        acc.append(list(logs['acc']))
        test_loss.append(list(logs['test_loss']))
        test_acc.append(list(logs['test_acc']))
        precision.append(list(logs['precision']))
        recall.append(list(logs['recall']))

    metric_names = ['loss', 'acc', 'test_loss', 'test_acc', 'precision', 'recall']

    for k, metric in enumerate([loss, acc, test_loss, test_acc, precision, recall]):
        graph_for_metric(args, metric, metric_names[k])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Palindrome model.')
    parser.add_argument('--name', type=str, default="test", help='name of the metric')
    parser.add_argument('--labels', nargs='+', help='legened labels')
    parser.add_argument('--num_exps', type=int, default=4, help='number of exps')

    args = parser.parse_args()

    generate_graph(args)