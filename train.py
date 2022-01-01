from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path
import hydra
import time
from src.model import Transformer
from src.data import get_dataset
from src.mask import *
from src.optimizer import get_optimizer
import tensorflow as tf
import os



@hydra.main(config_path="cfg", config_name='config')
def app(cfg):

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)
    
    @tf.function(experimental_relax_shapes=True)
    def train_step(inp, inp_len, tar, tar_len):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp_len, tar_len - 1, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp, 
                    True, 
                    enc_padding_mask, 
                    combined_mask, 
                    dec_padding_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)    
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_real, predictions)
    
    @tf.function(experimental_relax_shapes=True)
    def valid_step(inp, inp_len, tar, tar_len):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp_len, tar_len - 1, tar_inp)

        predictions, _ = transformer(
                inp,
                tar_inp,
                False,
                enc_padding_mask,
                combined_mask,
                dec_padding_mask
        )
        loss = loss_function(tar_real, predictions)

        valid_loss(loss)
        valid_accuracy(tar_real, predictions)

    train_ds, valid_ds, _, info = get_dataset(cfg)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    transformer = Transformer(cfg.model.num_layers, cfg.model.d_model, cfg.model.num_heads, cfg.model.dff,
            info['inp_vocab_size'], info['out_vocab_size'], info['inp_max_pe'], info['out_max_pe'])
    
    optimizer = get_optimizer(cfg.model.d_model)

    checkpoint_path = os.path.join(get_original_cwd(), "checkpoints/train")
    ckpt = tf.train.Checkpoint(transformer=transformer,
                            optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')
    
    for epoch in range(cfg.train.epoch):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        for (batch, data) in enumerate(train_ds):
            train_step(data['inp'], data['inp_len'], data['tar'], data['tar_len'])

            if batch % 50 == 0:
                print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                    ckpt_save_path))
        
        print ('[TRAIN] Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                train_loss.result(), 
                train_accuracy.result()))

        for (batch, data) in enumerate(valid_ds):
            valid_step(data['inp'], data['inp_len'], data['tar'], data['tar_len'])
        
        print ('[VALID] Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                valid_loss.result(), 
                valid_accuracy.result()))
        
        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


if __name__ == '__main__':
	app()
