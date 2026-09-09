# Build a Trainable CNN from Scratch in NumPy

Assemble a LeNet-style convolutional network entirely in NumPy, from numerically stable softmax and im2col-based convolutions all the way to an Adam-driven training loop. By the end you will have every layer, gradient, and optimizer wired into a working classifier you can train on synthetic images.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** argmax_rows
- [x] **2.** row_max
- [x] **3.** row_sum
- [x] **4.** exp_shifted
- [x] **5.** stable_softmax
- [x] **6.** one_hot
- [x] **7.** gather_true_class_probs
- [x] **8.** cross_entropy_loss
- [x] **9.** accuracy
- [x] **10.** he_std
- [x] **11.** he_init
- [x] **12.** init_zero_bias
- [ ] **13.** pad_2d
- [ ] **14.** output_spatial_size
- [ ] **15.** im2col
- [ ] **16.** col2im
- [ ] **17.** conv2d_forward
- [ ] **18.** conv2d_grad_input
- [ ] **19.** conv2d_grad_weights
- [ ] **20.** conv2d_grad_bias
- [ ] **21.** conv2d_backward
- [ ] **22.** maxpool2d_forward
- [ ] **23.** scatter_grad_window
- [ ] **24.** maxpool2d_backward
- [ ] **25.** relu_forward
- [ ] **26.** relu_backward
- [ ] **27.** flatten_forward
- [ ] **28.** flatten_backward
- [ ] **29.** linear_forward
- [ ] **30.** linear_grad_input
- [ ] **31.** linear_grad_weights
- [ ] **32.** linear_grad_bias
- [ ] **33.** linear_backward
- [ ] **34.** softmax_cross_entropy_forward
- [ ] **35.** softmax_cross_entropy_backward
- [ ] **36.** sgd_step
- [ ] **37.** adam_update_m
- [ ] **38.** adam_update_v
- [ ] **39.** adam_bias_correct
- [ ] **40.** adam_param_step
- [ ] **41.** adam_step
- [ ] **42.** init_conv_layer
- [ ] **43.** init_linear_layer
- [ ] **44.** init_lenet
- [ ] **45.** forward_conv_block
- [ ] **46.** forward_classifier_block
- [ ] **47.** lenet_forward
- [ ] **48.** backward_conv_block
- [ ] **49.** backward_classifier_block
- [ ] **50.** lenet_backward
- [ ] **51.** lenet_predict
- [ ] **52.** build_synthetic_image_dataset
- [ ] **53.** shuffle_indices
- [ ] **54.** train_test_split
- [ ] **55.** iterate_minibatches
- [ ] **56.** train_step
- [ ] **57.** train_one_epoch
- [ ] **58.** train_loop
- [ ] **59.** evaluate

---

Built on Deep-ML.
