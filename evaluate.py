from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path
import hydra
import time
from src.model import Transformer
from src.data import get_dataset, TOKEN_SOS, TOKEN_EOS
from src.mask import *
from src.optimizer import get_optimizer
import tensorflow as tf
import os



@hydra.main(config_path="cfg", config_name='config')
def app(cfg):

    _, _, test_ds, info = get_dataset(cfg)

    transformer = Transformer(cfg.model.num_layers, cfg.model.d_model, cfg.model.num_heads, cfg.model.dff,
            info['inp_vocab_size'], info['out_vocab_size'], info['inp_max_pe'], info['out_max_pe'])
    
    optimizer = get_optimizer(cfg.model.d_model)

    checkpoint_path = os.path.join(get_original_cwd(), "checkpoints/train")
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    assert ckpt_manager.latest_checkpoint
    print(ckpt_manager.latest_checkpoint)
    ckpt.restore(ckpt_manager.latest_checkpoint) # .expect_partial()

    sos_index = info['out_vocab'].index(TOKEN_SOS)
    eos_index = info['out_vocab'].index(TOKEN_EOS)
    correct_index = info['out_vocab'].index('Correct.')
    wrong_index = info['out_vocab'].index('Wrong.')
    its_index = info['out_vocab'].index('It"s')

    for (batch, data) in enumerate(test_ds.skip(1).take(10)):
        dec_input = [sos_index]
        output = tf.expand_dims(dec_input, 0)
        
        inp_np = data['inp'].numpy()
        print('Question:', end=' ')
        for i in range(inp_np.shape[1]):
            print(info['inp_vocab'][inp_np[0][i]], end=' ')
        print()

        tar_np = data['tar'].numpy()
        print('Ans:', end=' ')
        for i in range(tar_np.shape[1]):
            print(info['out_vocab'][tar_np[0][i]], end=' ')
        print()

        print('Prediction:', end=' ')
        for i in range(info['out_max_pe']):
            tar_len = tf.constant([i + 1], dtype=tf.int64)
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(data['inp_len'], tar_len, output)
            predictions, attention_weights = transformer(data['inp'], 
                    output,
                    False,
                    enc_padding_mask,
                    combined_mask,
                    dec_padding_mask)
            predictions = predictions[: ,-1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            
            if i == 0:
                if predictions.numpy()[0, 0, correct_index] > predictions.numpy()[0, 0, wrong_index]:
                    predicted_id = tf.constant([[correct_index]], dtype=tf.int32)
                else:
                    predicted_id = tf.constant([[wrong_index]], dtype=tf.int32)
            elif i == 1:
                predicted_id = tf.constant([[its_index]], dtype=tf.int32)

            output = tf.concat([output, predicted_id], axis=-1)

            print(info['out_vocab'][predicted_id.numpy()[0][0]], end=' ')
        print()

if __name__ == '__main__':
	app()
