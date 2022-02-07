import argparse
from os import path

DATA_DIR = path.join(path.dirname(path.dirname(__file__)), "data")
MODEL_DIR = path.join(DATA_DIR, "models")


def parse_args():
    parser = argparse.ArgumentParser('EDU segmentation toolkit 1.0')
    parser.add_argument('--prepare', action='store_true',
                        help='preprocess the RST-DT data and create the vocabulary')
    parser.add_argument('--train', action='store_true', help='train the segmentation model')
    parser.add_argument('--restore', action='store_true', help='continue training the segmentation model')
    parser.add_argument('--evaluate', action='store_true', help='evaluate the model')
    parser.add_argument('--segment', action='store_true', help='segment new files or input text')
    parser.add_argument('--gpu', type=str, help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam', help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=0.9, help='dropout rate')
    train_settings.add_argument('--ema_decay', type=float, default=0.95, help='exponential moving average decay')
    train_settings.add_argument('--max_grad_norm', type=float, default=5.0, help='clip gradients to this norm')
    train_settings.add_argument('--batch_size', type=int, default=32, help='batch size')
    train_settings.add_argument('--epochs', type=int, default=20, help='train epochs')
    train_settings.add_argument('--seed', type=int, default=123, help='the random seed')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--hidden_size', type=int, default=128, help='hidden size')
    model_settings.add_argument('--window_size', type=int, default=5, help='window size for restricted attention')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--rst_dir', default='../data/rst/',
                               help='the path of the rst data directory')
    path_settings.add_argument('--input_files', nargs='+',
                               help='list of files that contain the instances to segment')
    path_settings.add_argument('--model_dir', default=MODEL_DIR,
                               help='the dir to save the model')
    path_settings.add_argument('--result_dir', default='../data/results',
                               help='the directory to save edu segmentation results')
    path_settings.add_argument('--log_path', help='the file to output log')
    return parser.parse_args()
