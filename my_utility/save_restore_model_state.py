from keras.models import load_model
import numpy as np
from keras.engine.saving import load_weights_from_hdf5_group
import h5py
import json
from keras import optimizers


def restore_model_state(model, checkpoint_path):
    filepath = checkpoint_path+'_model_and_optimizer.h5'
    f = h5py.File(filepath, mode='r')
    load_weights_from_hdf5_group(f['model_weights'], model.layers)

    training_config = f.attrs.get('training_config')
    training_config = json.loads(training_config.decode('utf-8'))
    optimizer_config = training_config['optimizer_config']
    optimizer = optimizers.deserialize(optimizer_config)

    # model.compile(optimizer=loaded_model.optimizer,
    #               loss=loaded_model.loss, # the loss function has no state
    #               metrics=loaded_model.metrics,
    #               loss_weights=loaded_model.loss_weights,
    #               sample_weight_mode=loaded_model.sample_weight_mode)
    model.optimizer = optimizer
    other_configs = np.load(checkpoint_path+'_other_logs.npz')
    return other_configs['epoch'][0]


def save_model_state(model, checkpoint_path, epoch):
    model.save_weights(checkpoint_path+'_model_weights.h5')
    model.save(checkpoint_path+'_model_and_optimizer.h5')
    np.savez(checkpoint_path+'_other_logs.npz', epoch=[epoch])


def save_checkpoint_reduce_onplateau_callback(self, epoch, logs):
    data = np.zeros((11,))

    lr_schedular = self.callbacks[0]
    data[0] = lr_schedular.factor
    data[1] = lr_schedular.min_lr
    data[2] = lr_schedular.min_delta
    data[3] = lr_schedular.patience
    data[4] = lr_schedular.verbose
    data[5] = lr_schedular.cooldown
    data[6] = lr_schedular.cooldown_counter
    data[7] = lr_schedular.wait
    data[8] = lr_schedular.best

    if len(self.callbacks) > 1:
        chk_pt = self.callbacks[1]
        data[9] = chk_pt.best
        data[10] = chk_pt.epochs_since_last_save

    np.savez("callback_checkpoint.npz", data=data)