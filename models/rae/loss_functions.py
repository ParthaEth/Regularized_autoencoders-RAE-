import keras.backend as K


def per_pix_recon_loss(y_true, y_pred):
    reconstruction_loss = K.mean(K.sum(K.square(y_true - y_pred), axis=-1))
    return reconstruction_loss


def embeddig_loss(embedding):
    def _embedding_loss(y_true=0, y_pred=0):
        return K.mean(K.square(embedding), axis=[1])
    return _embedding_loss


def grad_pen_loss(embedding, entropy_qz):
    def _grad_pen_loss(y_true, y_pred):
        if entropy_qz is not None:
            loss = K.mean(K.square(entropy_qz*K.gradients(K.square(y_pred), embedding)))  # No batch shape is there so mean accross everything is ok
        else:
            loss = K.mean(K.square(K.gradients(K.square(y_pred), embedding)))
            print(loss)
        return loss
    return _grad_pen_loss


def total_loss(embedding, beta, apply_grad_pen, grad_pen_weight, recon_loss_func=None, entropy_qz=None,
               regularization_loss=None):
    def _total_loss(y_true, y_pred):
        if recon_loss_func == None:
            recon_loss = per_pix_recon_loss(y_true, y_pred)
        else:
            recon_loss = K.mean(recon_loss_func(y_true, y_pred), axis=[1, 2])
        _loss = recon_loss + beta * embeddig_loss(embedding)(0, 0)
        if apply_grad_pen:
            _loss += grad_pen_weight*grad_pen_loss(embedding, entropy_qz)(y_true, y_pred)
        if entropy_qz is not None:
            _loss -= beta * entropy_qz
        if regularization_loss is not None:
            _loss += regularization_loss
        return _loss
    return _total_loss