from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path
import hydra
import time
import matplotlib.pyplot as plt
from src.model import Transformer
from src.data import get_dataset, TOKEN_SOS, TOKEN_EOS
from src.mask import *
from src.optimizer import get_optimizer
import tensorflow as tf
import os


@hydra.main(config_path="cfg", config_name='config')
def app(cfg):

    @tf.function(experimental_relax_shapes=True)
    def eval(inp, inp_len, tar, tar_len):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp_len, tar_len, tar)
        predictions, attention_weights = transformer(inp, 
                tar,
                False,
                enc_padding_mask,
                combined_mask,
                dec_padding_mask)
        predictions = predictions[: ,-1:, :]
        return predictions, attention_weights

    def decode(inp, inp_vocab, tar, out, tar_vocab):
        dec_inp = []
        for i in range(inp.shape[0]):
            dec_inp.append([])
            for j in range(inp.shape[1]):
                dec_inp[-1].append(inp_vocab[inp[i, j]])

        dec_tar = []
        for i in range(tar.shape[0]):
            dec_tar.append([])
            for j in range(tar.shape[1]):
                dec_tar[-1].append(tar_vocab[tar[i, j]])

        dec_out = []
        for i in range(out.shape[0]):
            dec_out.append([])
            for j in range(out.shape[1]):
                dec_out[-1].append(tar_vocab[out[i, j]])

        return dec_inp, dec_tar, dec_out

    def calc_exact_match(tar, out, eos_index):
        cnt = 0
        wrong_index = []
        for i in range(tar.shape[0]):
            cnt += 1
            for j in range(tar.shape[1]):
                if tar[i, j] != out[i, j]:
                    cnt -= 1
                    wrong_index.append(i)
                    break
                if tar[i, j] == eos_index:
                    break
        return cnt, wrong_index

    _, _, test_ds, info = get_dataset(cfg)

    transformer = Transformer(cfg.model.num_layers, cfg.model.d_model, cfg.model.num_heads, cfg.model.dff,
            info['inp_vocab_size'], info['out_vocab_size'], info['inp_max_pe'], info['out_max_pe'])
    
    optimizer = get_optimizer(cfg.model.d_model)

    checkpoint_path = os.path.join(get_original_cwd(), "checkpoints/train")
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    assert ckpt_manager.latest_checkpoint
    print(ckpt_manager.latest_checkpoint)
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

    sos_index = info['out_vocab'].index(TOKEN_SOS)
    eos_index = info['out_vocab'].index(TOKEN_EOS)
    correct_index = info['out_vocab'].index('Correct.')
    wrong_index = info['out_vocab'].index('Wrong.')
    its_index = info['out_vocab'].index('It"s')

    total_cnt = 0
    correct_cnt = 0
    exact_cnt = 0
    wrong_results = []

    for data in test_ds:
        dec_input = [sos_index]
        output = tf.tile(tf.expand_dims(dec_input, 0), [tf.shape(data['inp_len'])[0], 1])

        for i in range(info['out_max_pe']):
            tar_len = tf.tile(tf.constant([i + 1], dtype=tf.int64), [tf.shape(data['inp_len'])[0]])
            predictions, _ = eval(data['inp'], data['inp_len'], output, tar_len)
            predictions = predictions.numpy()
            if i == 0:
                correct_pred = predictions[:, 0, correct_index:correct_index+1]
                wrong_pred = predictions[:, 0, wrong_index:wrong_index+1]
                predicted_id = np.where(
                        correct_pred > wrong_pred,
                        np.ones_like(correct_pred, dtype=np.int32) * correct_index,
                        np.ones_like(correct_pred, dtype=np.int32) * wrong_index
                )
            elif i == 1:
                predicted_id = np.ones((predictions.shape[0], 1)) * its_index
            else:
                predicted_id = np.argmax(predictions, axis=-1)
            predicted_id = tf.constant(predicted_id, dtype=tf.int32)
            output = tf.concat([output, predicted_id], axis=-1)
        
        
        total_cnt += data['inp_len'].numpy().shape[0]
        correct_cnt += int(tf.reduce_sum(tf.cast(tf.cast(data['tar'][:, 1], tf.int32) == output[:, 1], tf.int32)).numpy())
        _exact_cnt, wrong_indexes = calc_exact_match(data['tar'].numpy(), output.numpy(), eos_index)
        exact_cnt += _exact_cnt
        if wrong_index:
            dec_inp, dec_tar, dec_out = decode(data['inp'].numpy(), info['inp_vocab'], data['tar'].numpy(), output, info['out_vocab'])
            for index in wrong_indexes:
                wrong_results.append((dec_inp[index], dec_tar[index], dec_out[index]))

    print('Accuracy of Correctness:', correct_cnt / total_cnt)
    print('Accuracy of Exact Match:', exact_cnt / total_cnt)
    print(wrong_results[:10])

    # Visualize Attention Weights
    os.makedirs(os.path.join(get_original_cwd(), cfg.eval.eval_dir), exist_ok=True)

    for data in test_ds.take(1):
        dec_input = [sos_index]
        output = tf.tile(tf.expand_dims(dec_input, 0), [tf.shape(data['inp_len'])[0], 1])

        for i in range(info['out_max_pe']):
            tar_len = tf.tile(tf.constant([i + 1], dtype=tf.int64), [tf.shape(data['inp_len'])[0]])
            predictions, attention_weights = eval(data['inp'], data['inp_len'], output, tar_len)
            predictions = predictions.numpy()
            if i == 0:
                correct_pred = predictions[:, 0, correct_index:correct_index+1]
                wrong_pred = predictions[:, 0, wrong_index:wrong_index+1]
                predicted_id = np.where(
                        correct_pred > wrong_pred,
                        np.ones_like(correct_pred, dtype=np.int32) * correct_index,
                        np.ones_like(correct_pred, dtype=np.int32) * wrong_index
                )
            elif i == 1:
                predicted_id = np.ones((predictions.shape[0], 1)) * its_index
            else:
                predicted_id = np.argmax(predictions, axis=-1)
            predicted_id = tf.constant(predicted_id, dtype=tf.int32)
            output = tf.concat([output, predicted_id], axis=-1)
        
        dec_inp, dec_tar, dec_out = decode(data['inp'].numpy(), info['inp_vocab'], data['tar'].numpy(), output, info['out_vocab'])
        for k in range(5):
            with open(os.path.join(get_original_cwd(), cfg.eval.eval_dir, f'{k}.txt'), 'w') as f:
                print(' '.join(dec_inp[k]), file=f)
                print(' '.join(dec_tar[k]), file=f)
                print(' '.join(dec_out[k]), file=f)
            for j in range(cfg.model.num_layers):
                atten = attention_weights['decoder_layer{}_block1'.format(j+1)].numpy()[k, 0, :, :]
                plt.matshow(atten)
                plt.xticks([m for m in range(info['out_max_pe'])], dec_out[k][:-1])
                plt.yticks([m for m in range(info['out_max_pe'])], dec_out[k][1:])
                plt.savefig(os.path.join(get_original_cwd(), cfg.eval.eval_dir, f'{k}_head0_block1_layer{j}.jpg'), dpi=300)

                atten = attention_weights['decoder_layer{}_block2'.format(j+1)].numpy()[k, 0, :, :]
                plt.matshow(atten)
                plt.xticks([m for m in range(data['inp'].numpy().shape[1])], dec_inp[k])
                plt.yticks([m for m in range(info['out_max_pe'])], dec_out[k][1:])
                plt.savefig(os.path.join(get_original_cwd(), cfg.eval.eval_dir, f'{k}_head0_block2_layer{j}.jpg'), dpi=300)


if __name__ == '__main__':
	app()
