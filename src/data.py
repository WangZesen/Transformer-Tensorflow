import random
import tensorflow as tf
from tensorflow_text import WhitespaceTokenizer
from hydra.utils import to_absolute_path

TOKEN_SOS = 'SOS'
TOKEN_EOS = 'EOS'
TOKEN_PAD = 'PAD'

def get_vocab(data_dirs):
    all_words_question = []
    all_words_answer = []
    max_pe_question = 0
    max_pe_answer = 0
    for data_dir in data_dirs:
        with open(data_dir, 'r') as f:
            line = f.readline()
            while line:
                question = line.strip('\n').split('\t')[0].split(' ')
                all_words_question.extend(question)
                answer = line.strip('\n').split('\t')[1].split(' ')
                all_words_answer.extend(answer)
                max_pe_question = max(max_pe_question, len(question))
                max_pe_answer = max(max_pe_answer, len(answer) + 2) # Include EOS and SOS
                line = f.readline()
    question_vocab = [TOKEN_PAD] + sorted(list(set(all_words_question)))
    answer_vocab = [TOKEN_PAD] + sorted(list(set(all_words_answer)))
    answer_vocab.extend([TOKEN_SOS, TOKEN_EOS])
    return question_vocab, answer_vocab, max_pe_question, max_pe_answer


def get_lookup_table(vocab):
    init = tf.lookup.KeyValueTensorInitializer(
        keys=vocab,
        values=tf.constant([i for i in range(len(vocab))], dtype=tf.int64)
    )
    table = tf.lookup.StaticVocabularyTable(
        init,
        num_oov_buckets=1
    )
    return table


def get_raw_data(data_dir):
    data = []
    with open(data_dir, 'r') as f:
        line = f.readline()
        while line:
            question = line.strip('\n').split('\t')[0]
            answer = f'{TOKEN_SOS} ' + line.strip('\n').split('\t')[1] + f' {TOKEN_EOS}'
            data.append((question, answer))
            line = f.readline()
    return data


def get_dataset(cfg):
    data_dirs = [to_absolute_path(cfg.data.train_data.dir), to_absolute_path(cfg.data.test_data.dir)]
    inp_vocab, out_vocab, inp_max_pe, out_max_pe = get_vocab(data_dirs)
    inp_table = get_lookup_table(inp_vocab)
    out_table = get_lookup_table(out_vocab)
    tokenizer = WhitespaceTokenizer()

    def _parse_fn(data):
        inp_token = inp_table[tokenizer.tokenize(data[0])]
        inp_len = tf.shape(inp_token)[0]
        out_token = out_table[tokenizer.tokenize(data[1])]
        out_len = tf.shape(out_token)[0]
        return {
            'inp': inp_token,
            'inp_len': inp_len,
            'tar': out_token,
            'tar_len': out_len,
        }
    
    def _get_tf_dataset(raw_data, batch_size=None):
        return tf.data.Dataset.from_tensor_slices(raw_data[:round(len(raw_data) * cfg.data.train_split)]) \
            .map(_parse_fn) \
            .cache() \
            .shuffle(cfg.data.buffer_size, seed=cfg.data.random_seed, reshuffle_each_iteration=True) \
            .padded_batch(batch_size if batch_size else cfg.train.batch_size)

    raw_data = get_raw_data(data_dirs[0])
    random.seed(cfg.data.random_seed)
    random.shuffle(raw_data)
    train_ds = _get_tf_dataset(raw_data[:round(len(raw_data) * cfg.data.train_split)])
    valid_ds = _get_tf_dataset(raw_data[round(len(raw_data) * cfg.data.train_split):])

    raw_data = get_raw_data(data_dirs[1])
    test_ds = _get_tf_dataset(raw_data, 1)

    info = {
        'inp_vocab_size': len(inp_vocab),
        'out_vocab_size': len(out_vocab),
        'inp_vocab': inp_vocab,
        'out_vocab': out_vocab,
        'inp_max_pe': inp_max_pe,
        'out_max_pe': out_max_pe,
    }
    
    return train_ds, valid_ds, test_ds, info
