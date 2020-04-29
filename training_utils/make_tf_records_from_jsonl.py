import argparse
import fastBPE
import glob
import numpy as np
import os
import pdb
import platform
import re
import sys
import tensorflow as tf
import time
from tqdm import tqdm

CODES_FILE = '../codes'
VOCAB_FILE = '../vocab'


def numericalize(tokens, word2idx):
    count = 0
    for x in tokens:
        if x not in word2idx:
            print(x)
            count += 1
    return count > 1, [word2idx.get(x, word2idx['<unk>']) for x in tokens]


def preprocess_text(tokenized_train_text: str):
    tokenized_train_text = re.findall(r'\S+|\n', tokenized_train_text)
    tokenized_train_text = list(filter(lambda x: x != u'@@', tokenized_train_text))
    return tokenized_train_text


def load_examples(path_to_train_file, vocab, seq_len):
    train_examples = [json.loads(line) for line in open(path_to_train_file)]
    train_codes, train_corpus = zip(*[(example['control_code'], example['text']) for example in train_examples])

    bpe = fastBPE.fastBPE(CODES_FILE, VOCAB_FILE)
    tokenized_train_corpus = bpe.apply(train_corpus)
    tokenized_train_corpus = [preprocess_text(tokenized_train_text) for tokenized_train_text in tokenized_train_corpus]

    examples = []
    for control_code, text in zip(train_codes, tokenized_train_corpus):
        if control_code not in vocab:
            raise RuntimeError(f'{control_code} not in vocab')

        for i in range(0, len(text), seq_len):
            text_chunk = text[i: i + seq_len]
            if text_chunk != seq_len:
                break
            examples.append((control_code, text_chunk))

    return examples


def load_vocab():
    # load the vocabulary from file
    vocab = open('../vocab', encoding='utf-8').read().split('\n')
    vocab = list(map(lambda x: x.split(' ')[0], vocab)) + ['<unk>'] + ['\n']
    print('{} unique words'.format(len(vocab)))
    return vocab


def main():
    parser = argparse.ArgumentParser(description='TensorFlow code for creating TFRecords data')
    parser.add_argument('--jsonl_file', type=str, required=True,
                        help='location of jsonl file to convert to TFRecords')
    parser.add_argument('--sequence_len', type=int, required=True,
                        help='sequence length of model being fine-tuned (256 or 512)')
    args = parser.parse_args()

    # Load vocab
    vocab = load_vocab()
    examples = load_examples(args.text_file, vocab, args.sequence_len)

    # Creating a mapping from unique characters to indices
    word2idx = {u: i for i, u in enumerate(vocab)}

    tfrecords_fname = args.text_file.lower() + '.tfrecords'

    total = 0
    skipped = 0
    with tf.io.TFRecordWriter(tfrecords_fname) as writer:
        for control_code, tokenized_text in tqdm(examples):
            flag_input, inputs = numericalize([control_code] + tokenized_text[:-1], word2idx)
            flag_output, outputs = numericalize(tokenized_text, word2idx)
            total += 1
            if flag_input or flag_output:
                skipped += 1
                continue

            if len(inputs) != seq_length + 1 or len(outputs) != seq_length + 1:
                break

            features = tf.train.Features(
                feature={
                    'input': tf.train.Feature(int64_list=tf.train.Int64List(value=inputs)),
                    'output': tf.train.Feature(int64_list=tf.train.Int64List(value=outputs))
                }
            )
            example_proto = tf.train.Example(features=features)
            writer.write(example_proto.SerializeToString())
    print('Done')
    print('Skipped', skipped, 'of', total)


if __name__ == '__main__':
    main()
