import glob
import logging
import os
import random
import subprocess

import tensorflow as tf

from neuraleduseg.config import parse_args
from neuraleduseg.model.atten_seg import AttnSegModel
from neuraleduseg.preprocess import preprocess_rst_data
from neuraleduseg.rst_edu_reader import RSTData


def prepare(args):
    logger = logging.getLogger('SegEDU')
    logger.info('Randomly sample 10% of the training data for validation...')
    raw_train_dir = os.path.join(args.rst_dir, 'TRAINING')
    raw_dev_dir = os.path.join(args.rst_dir, 'DEV')

    if not os.path.exists(raw_dev_dir):
        os.makedirs(raw_dev_dir)
    raw_train_doc_ids = [file.split('.')[0] for file in os.listdir(raw_train_dir) if file.endswith('.out')]
    random.shuffle(raw_train_doc_ids)
    dev_doc_ids = raw_train_doc_ids[: int(len(raw_train_doc_ids) * 0.1)]
    for doc_id in dev_doc_ids:
        p = subprocess.call('mv {}/{}* {}'.format(raw_train_dir, doc_id, raw_dev_dir), shell=True)

    preprocessed_train_dir = os.path.join(args.rst_dir, 'preprocessed/train/')
    preprocessed_dev_dir = os.path.join(args.rst_dir, 'preprocessed/dev/')
    preprocessed_test_dir = os.path.join(args.rst_dir, 'preprocessed/test/')
    logger.info('Preprocessing Train data...')
    preprocess_rst_data(os.path.join(args.rst_dir, 'TRAINING'), preprocessed_train_dir)
    logger.info('Preprocessing Dev data...')
    preprocess_rst_data(os.path.join(args.rst_dir, 'DEV'), preprocessed_dev_dir)
    logger.info('Preprocessing Test data...')
    preprocess_rst_data(os.path.join(args.rst_dir, 'TEST'), preprocessed_test_dir)


def train(args):
    logger = logging.getLogger('SegEDU')
    logger.info('Loading data...')
    train_files = glob.glob(os.path.join(args.rst_dir, 'preprocessed/train', '*.preprocessed'))
    dev_files = glob.glob(os.path.join(args.rst_dir, 'preprocessed/dev', '*.preprocessed'))
    test_files = glob.glob(os.path.join(args.rst_dir, 'preprocessed/test', '*.preprocessed'))
    rst_data = RSTData(train_files=train_files, dev_files=dev_files, test_files=test_files)
    logger.info('Number of training batches... {}'.format(len(rst_data.train_samples) // args.batch_size))
    logger.info('Initialize the model...')
    model = AttnSegModel(args)
    if args.restore:
        model.restore('best', args.model_dir)
    logger.info('Training the model...')
    model.train(rst_data, args.epochs, args.batch_size, print_every_n_batch=20)
    logger.info('Done with model training')


def evaluate(args):
    logger = logging.getLogger('SegEDU')
    logger.info('Loading data...')
    test_files = glob.glob(os.path.join(args.rst_dir, 'preprocessed/test', '*.preprocessed'))
    rst_data = RSTData(test_files=test_files)
    logger.info('Loading the model...')
    model = AttnSegModel(args)
    model.restore('best', args.model_dir)
    eval_batches = rst_data.gen_mini_batches(args.batch_size, test=True, shuffle=False)
    perf = model.evaluate(eval_batches, print_result=False)
    logger.info(perf)


def segment(args):
    """
    Segment raw text into edus.
    """
    import spacy

    logger = logging.getLogger('SegEDU')
    rst_data = RSTData()
    logger.info('Loading the model...')
    model = AttnSegModel(args)
    model.restore('best', args.model_dir)
    if model.use_ema:
        model.sess.run(model.ema_backup_op)
        model.sess.run(model.ema_assign_op)

    spacy_nlp = spacy.load('en', disable=['parser', 'ner', 'textcat'])
    for file in args.input_files:
        logger.info('Segmenting {}...'.format(file))
        raw_sents = []
        with open(file, 'r') as fin:
            for line in fin:
                line = line.strip()
                if line:
                    raw_sents.append(line)
        samples = []
        for sent in spacy_nlp.pipe(raw_sents, batch_size=1000, n_threads=5):
            samples.append({'words': [token.text for token in sent],
                            'edu_seg_indices': []})
        rst_data.test_samples = samples
        data_batches = rst_data.gen_mini_batches(args.batch_size, test=True, shuffle=False)

        edus = []
        for batch in data_batches:
            batch_pred_segs = model.segment(batch)
            for sample, pred_segs in zip(batch['raw_data'], batch_pred_segs):
                one_edu_words = []
                for word_idx, word in enumerate(sample['words']):
                    if word_idx in pred_segs:
                        edus.append(' '.join(one_edu_words))
                        one_edu_words = []
                    one_edu_words.append(word)
                if one_edu_words:
                    edus.append(' '.join(one_edu_words))

        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
        save_path = os.path.join(args.result_dir, os.path.basename(file))
        logger.info('Saving into {}'.format(save_path))
        with open(save_path, 'w') as fout:
            for edu in edus:
                fout.write(edu + '\n')


def main():
    logging.basicConfig(level=logging.INFO)
    tf.compat.v1.logging.set_verbosity('ERROR')

    args = parse_args()
    logger = logging.getLogger("SegEDU")
    logger.info('Running with args : {}'.format(args))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.prepare:
        prepare(args)
    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)
    if args.segment:
        segment(args)


if __name__ == '__main__':
    main()
