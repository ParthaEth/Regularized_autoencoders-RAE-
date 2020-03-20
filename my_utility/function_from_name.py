from keras.losses import mae, mse, binary_crossentropy
import numpy as np


def get_recon_loss_func(loss_func_name, weights=None):
    """Returns reconstruction loss given its name. Seperate names with + to combine different losses"""
    # loss_func_name: string specifying loss function names e.g. L1+L2 represents weights sum of l1 and l2 loss
    # weights: optinally provide a numpy array of floating point values to weigh the losses differently
    if loss_func_name is None:
        return None

    loss_func_list = loss_func_name.split('+')
    if weights is not None and len(weights) != len(loss_func_list):
        raise ValueError('Length of weights must be same as number of loss functions')
    if weights is None:
        weights = np.ones(len(loss_func_list), 'float32')

    def total_recon_loss(y_true, y_pred):
        tot_recon_loss = 0
        for idx, loss_name in enumerate(loss_func_list):
            if loss_name.upper() == 'L1' or loss_name.upper() == 'MAE':
                tot_recon_loss += weights[idx] * mae(y_true, y_pred)
            elif loss_name.upper() == 'L2' or loss_name.upper() == 'MSE':
                tot_recon_loss += weights[idx] * mse(y_true, y_pred)
            elif loss_name.upper() == 'BINARY_CROSSENTROPY' or loss_name.upper() == 'BINARY_CROSS_ENTROPY':
                tot_recon_loss += weights[idx] * binary_crossentropy(y_true, y_pred)
            else:
                raise NotImplementedError(str(loss_name) + " has no implementation")
        return tot_recon_loss

    return total_recon_loss
