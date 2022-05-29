# coding=utf-8
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath',
                        type=str,
                        help='path to dataset')

    parser.add_argument('--fileModelSave',
                        type=str,
                        help='path to save model')
    

    parser.add_argument('--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=100)
    parser.add_argument('--train_batch_size',
                        type=int,
                        help='number of examples in one batch during train',
                        default=5)
    parser.add_argument('--test_batch_size',
                        type=int,
                        help='number of examples in one batch during test',
                        default=5)

    parser.add_argument('--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.001)
    parser.add_argument('--numDevice',
                        type=int,
                        help='id of gpu ',
                        default=0)

    parser.add_argument('--warmup_steps',
                        type=int,
                        help='num of warmup_steps',
                        default=100)

    parser.add_argument('--weight_decay',
                        type=float,
                        help='ratio of decay',
                        default=0.2)

    parser.add_argument('--dropout_rate',
                        type=float,
                        help='ratio of dropout',
                        default=0.1)
 
    return parser
