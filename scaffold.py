"""
Build a Trainable CNN from Scratch in NumPy scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Scaffold demo for a NumPy LeNet-style CNN trained end-to-end."""

import numpy as np

from solution import (
    argmax_rows,
    row_max,
    row_sum,
    exp_shifted,
    stable_softmax,
    one_hot,
    gather_true_class_probs,
    cross_entropy_loss,
    accuracy,
    he_std,
    he_init,
    init_zero_bias,
    pad_2d,
    output_spatial_size,
    im2col,
    col2im,
    conv2d_forward,
    conv2d_grad_input,
    conv2d_grad_weights,
    conv2d_grad_bias,
    conv2d_backward,
    maxpool2d_forward,
    scatter_grad_window,
    maxpool2d_backward,
    relu_forward,
    relu_backward,
    flatten_forward,
    flatten_backward,
    linear_forward,
    linear_grad_input,
    linear_grad_weights,
    linear_grad_bias,
    linear_backward,
    softmax_cross_entropy_forward,
    softmax_cross_entropy_backward,
    sgd_step,
    adam_update_m,
    adam_update_v,
    adam_bias_correct,
    adam_param_step,
    adam_step,
    init_conv_layer,
    init_linear_layer,
    init_lenet,
    forward_conv_block,
    forward_classifier_block,
    lenet_forward,
    backward_conv_block,
    backward_classifier_block,
    lenet_backward,
    lenet_predict,
    build_synthetic_image_dataset,
    shuffle_indices,
    train_test_split,
    iterate_minibatches,
    train_step,
    train_one_epoch,
    train_loop,
    evaluate,
)


def _print_param_shapes(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _print_param_shapes(v, prefix=f"{prefix}{k}.")
    elif isinstance(obj, np.ndarray):
        print(f"  {prefix[:-1]}: shape={obj.shape}")
    else:
        print(f"  {prefix[:-1]}: {type(obj).__name__}")


if __name__ == "__main__":
    np.random.seed(0)

    # ---- Dataset ---------------------------------------------------------
    num_samples = 64
    num_classes = 3
    image_size = 28
    in_channels = 1

    x, y = build_synthetic_image_dataset(
        num_samples=num_samples,
        num_classes=num_classes,
        image_size=image_size,
        in_channels=in_channels,
        seed=0,
    )
    print(f"Dataset shapes: x={x.shape}, y={y.shape}")
    print(f"Label distribution: {np.bincount(y, minlength=num_classes)}")

    x_train, y_train, x_test, y_test = train_test_split(x, y, test_fraction=0.25, seed=0)
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    # ---- Model init ------------------------------------------------------
    params = init_lenet(in_channels=in_channels, num_classes=num_classes, seed=0)
    print("Parameter tensors:")
    _print_param_shapes(params)

    # ---- One forward pass before training -------------------------------
    logits0, _ = lenet_forward(x_train[:8], params)
    probs0 = stable_softmax(logits0)
    init_loss = cross_entropy_loss(probs0, y_train[:8])
    init_acc = accuracy(logits0, y_train[:8])
    print(f"Initial mini-batch loss: {init_loss:.4f}, accuracy: {init_acc:.3f}")

    # ---- Train -----------------------------------------------------------
    params, loss_history = train_loop(
        params,
        x_train,
        y_train,
        num_epochs=3,
        batch_size=16,
        lr=1e-3,
        beta_one=0.9,
        beta_two=0.999,
        eps=1e-8,
        seed=0,
    )
    print(f"Training steps: {len(loss_history)}")
    print(f"First loss: {loss_history[0]:.4f}, last loss: {loss_history[-1]:.4f}")

    # ---- Evaluate --------------------------------------------------------
    train_acc = evaluate(params, x_train, y_train)
    test_acc = evaluate(params, x_test, y_test)
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test  accuracy: {test_acc:.3f}")

    # ---- Sample predictions ---------------------------------------------
    preds = lenet_predict(x_test[:8], params)
    print(f"Sample predictions: {preds.tolist()}")
    print(f"Sample labels:      {y_test[:8].tolist()}")
