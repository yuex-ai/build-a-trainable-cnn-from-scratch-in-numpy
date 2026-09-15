"""
Build a Trainable CNN from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - argmax_rows
def argmax_rows(matrix):
    # TODO: return the index of the largest element in each row of a 2D array
    '''m,n=matrix.shape
    argmax=[]
    for i in range(m):
        max_value=matrix[i,0]
        max_index=0
        for j in range(1,n):
            if matrix[i,j]>=max_value:
                max_value=matrix[i,j]
                max_index=j
        argmax.append(max_index)
    return argmax'''
    return np.argmax(matrix, axis=1)

# Step 2 - row_max
import numpy as np

def row_max(matrix):
    # TODO: return the maximum value of each row of `matrix` with keepdims True for broadcasting.
    return np.max(matrix,axis=1,keepdims=True)

# Step 3 - row_sum
import numpy as np

def row_sum(matrix):
    """Return per-row sums of a 2D array with shape (N, 1)."""
    # TODO: return the sum along axis 1 keeping the reduced dimension
    return np.sum(matrix,axis=1,keepdims=True)

# Step 4 - exp_shifted
import numpy as np

def exp_shifted(logits):
    """Subtract per-row max from logits and exponentiate elementwise."""
    # TODO: shift each row of logits by its max and return elementwise exp
    max_logits=np.max(logits,axis=1,keepdims=True)
    
    return np.exp(logits-max_logits)

# Step 5 - stable_softmax
def stable_softmax(logits):
    # TODO: Compute a numerically stable softmax row-wise over (N, C) logits.
    logits_max=np.max(logits,axis=1,keepdims=True)
    logits_exp=np.exp(logits-logits_max)
    softmax=logits_exp/np.sum(logits_exp,axis=1,keepdims=True)
    return softmax

# Step 6 - one_hot
def one_hot(labels, num_classes):
    # TODO: convert integer labels into a (N, num_classes) one-hot float matrix
    N=len(labels)
    initial=np.zeros((N,num_classes),dtype=np.float32)

    for i in range(N):
        label=labels[i]
        initial[i,label]=1.0
    return initial

# Step 7 - gather_true_class_probs
def gather_true_class_probs(probs, labels):
    # TODO: return probs[i, labels[i]] for every row i as a 1D length-N array.
    N=probs.shape[0]
    result=probs[np.arange(N),labels]
    return result

# Step 8 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, labels, eps=1e-12):
    # TODO: return the mean negative log-likelihood of the true-class probabilities
    N=probs.shape[0]
    correct_p=probs[np.arange(N),labels]
    correct_p=np.clip(correct_p,eps,1)
    p_log=np.log(correct_p)
    p_log_mean=-np.mean(p_log)
    return p_log_mean

# Step 9 - accuracy
def accuracy(logits_or_probs, labels):
    # TODO: return the fraction of rows whose argmax matches the integer label.
    N=logits_or_probs.shape[0]
    preds=[]
    for i in range(N):
        max_index=np.argmax(logits_or_probs[i])
        if max_index==labels[i]:
            preds.append(1)
        else:
            preds.append(0)
    acc=1/N*(sum(preds))
    return acc

# Step 10 - he_std
def he_std(fan_in):
    # TODO: return the He initialization standard deviation sqrt(2 / fan_in).
    return round(np.sqrt(2/fan_in),6)

# Step 11 - he_init
def he_init(shape, fan_in, seed):
    # TODO: sample a weight tensor from a normal distribution scaled by He std using the seed.
    np.random.seed(seed)
    std=np.sqrt(2/fan_in)
    init=np.random.normal(loc=0,scale=std,size=shape)
    return init

# Step 12 - init_zero_bias
import numpy as np

def init_zero_bias(length):
    # TODO: return a 1D float array of zeros with the given length.
    return np.zeros(length,dtype=np.float64)

# Step 13 - pad_2d
def pad_2d(images, pad):
    # TODO: zero-pad the spatial (H, W) dims of a 4D (N, C, H, W) tensor by `pad` on each side.
    N,C,H,W=images.shape
    padding=np.zeros((N,C,H+2*pad,W+2*pad),dtype=images.dtype)
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    padding[n][c][h+pad][w+pad]=images[n][c][h][w]
    return padding

# Step 14 - output_spatial_size
def output_spatial_size(input_size, kernel, stride, padding):
    # TODO: return the conv/pool output spatial dimension from input_size, kernel, stride, padding
    return (input_size-kernel+padding*2)//stride+1

# Step 15 - im2col
def im2col(images, kernel_h, kernel_w, stride, padding):
    # TODO: Unroll overlapping patches of a 4D image tensor into a 2D column matrix.
    N,C,H,W=images.shape
    output_h = (H - kernel_h + 2 * padding) // stride + 1
    output_w = (W - kernel_w + 2 * padding) // stride + 1
    padded=pad_2d(images,padding)
    num_rows = N * output_h*output_w
    num_cols = C * kernel_h*kernel_w
    result = np.zeros((num_rows, num_cols),dtype=padded.dtype)
    row_idx=0
    for n in range(N):
        for h_out in range(output_h):
            for w_out in range (output_w):
                h_start=h_out*stride
                w_start=w_out*stride
                output_n=padded[n,:,h_start:h_start+kernel_h,w_start:w_start+kernel_w]
                result[row_idx,:]=output_n.reshape(-1)
                row_idx+=1
    return result

# Step 16 - col2im
def col2im(cols, input_shape, kernel_h, kernel_w, stride, padding):
    # TODO: re-roll a (N*out_h*out_w, C*kh*kw) column matrix back into a (N, C, H, W) tensor
    N, C, H, W = input_shape
    output_h = (H - kernel_h + padding * 2) // stride + 1
    output_w = (W - kernel_w + padding * 2) // stride + 1
    padded_H=H+2*padding
    padded_W=W+2*padding
    grad_padded=np.zeros((N,C,padded_H,padded_W),dtype=cols.dtype)
    idx=0
    for n in range(N):
        for h_out in range(output_h):
            for w_out in range(output_w):
                h_start=h_out*stride
                w_start=w_out*stride
                patch_flat=cols[idx,:]
                patch=patch_flat.reshape(C,kernel_h,kernel_w)
                grad_padded[n,:,h_start:h_start+kernel_h,w_start:w_start+kernel_w]+=patch
                idx+=1
    grad=grad_padded[:,:,padding:padding+H,padding:padding+W]
    return grad

# Step 17 - conv2d_forward
def conv2d_forward(x, weights, bias, stride, padding):
    # TODO: convolve x with weights using im2col, add bias, return output and a backprop cache.
    N, C, H, W = x.shape
    _,_,kernel_h,kernel_w=weights.shape
    out_h = (H - kernel_h + 2 * padding) // stride + 1
    out_w = (W - kernel_w + 2 * padding) // stride + 1
    out_channels=weights.shape[0]
    cols=im2col(x,kernel_h,kernel_w,stride,padding)
    w_shaped=weights.reshape(out_channels,-1)
    y=(cols@w_shaped.T)+bias
    y=y.reshape(N,out_h,out_w,out_channels)
    y=y.transpose(0,3,1,2)
    cache={
        'x_shape':x.shape,
        'weights':weights,
        'cols':cols,
        'stride':stride,
        'padding':padding,
        'kernel_h':kernel_h,
        'kernel_w':kernel_w
    }
    return y,cache

# Step 18 - conv2d_grad_input
def conv2d_grad_input(d_out, cache):
    # TODO: backprop d_out through the conv input using col2im
    x_shape = cache['x_shape']        # (N, C_in, H, W)
    weights = cache['weights']        # (C_out, C_in, kh, kw)
    stride = cache['stride']
    padding = cache['padding']
    kernel_h = cache['kernel_h']
    kernel_w = cache['kernel_w']
    N,C_out,out_h,out_w=d_out.shape
    _,C_in,H,W=x_shape
    d_out_transpose=d_out.transpose(0,2,3,1)
    d_out_row=d_out_transpose.reshape(N*out_h*out_w,C_out)
    W_row=weights.reshape(C_out,-1)
    d_cols=d_out_row @ W_row
    d_x=col2im(d_cols,x_shape,kernel_h,kernel_w,stride,padding)
    return d_x

# Step 19 - conv2d_grad_weights
def conv2d_grad_weights(d_out, cache):
    # TODO: return dL/dW shaped (C_out, C_in, kH, kW) from d_out and the im2col cache.
    cols=cache['cols']
    weights=cache['weights']
    kernel_h=cache['kernel_h']
    kernel_w=cache['kernel_w']
    C_out,C_in,_,_=weights.shape
    N,_,out_h,out_w=d_out.shape
    d_out_row=d_out.transpose(0,2,3,1).reshape(-1,C_out)
    dw_row=d_out_row.T@cols
    d_w=dw_row.reshape(C_out,C_in,kernel_h,kernel_w)
    return d_w

# Step 20 - conv2d_grad_bias
def conv2d_grad_bias(d_out):
    # TODO: return a length C_out gradient by reducing d_out over batch and spatial axes
    d_b=np.sum(d_out,axis=(0,2,3))
    return d_b

# Step 21 - conv2d_backward
def conv2d_backward(d_out, cache):
    # TODO: return (dx, dW, db) using the conv2d gradient helpers and the forward cache
    dx = conv2d_grad_input(d_out, cache)       # (N, C_in, H, W)
    dW = conv2d_grad_weights(d_out, cache)     # (C_out, C_in, kH, kW)
    db = conv2d_grad_bias(d_out)            # (C_out,)

    return dx, dW, db

# Step 22 - maxpool2d_forward
def maxpool2d_forward(x, kernel, stride):
    # TODO: run 2D max pooling and cache the in-window argmax of each output cell.
    N,C,H,W=x.shape
    out_h=(H-kernel)//stride+1
    out_w=(W-kernel)//stride+1
    value=np.zeros((N,C,out_h,out_w),dtype=x.dtype)
    argmax_index=np.zeros((N,C,out_h,out_w),dtype=int)
    for n in range(N):
        for c in range(C):
            for h in range(out_h):
                for w in range(out_w):
                    h_start=h*stride
                    w_start=w*stride
                    value[n,c,h,w]=np.max(x[n,c,h_start:h_start+kernel,w_start:w_start+kernel])
                    argmax_index[n,c,h,w]=np.argmax(x[n,c,h_start:h_start+kernel,w_start:w_start+kernel].flatten())
    cache={
        'x_shape':x.shape,
        'argmax':argmax_index,
        'kernel':kernel,
        'stride':stride
    }
    return value,cache

# Step 23 - scatter_grad_window
import numpy as np

def scatter_grad_window(grad_value, argmax_index, kernel):
    # TODO: place grad_value at the argmax position within a (kernel, kernel) zero array.
    matrix=np.zeros((kernel,kernel))
    matrix[argmax_index//kernel,argmax_index%kernel]=grad_value
    return matrix

# Step 24 - maxpool2d_backward
def maxpool2d_backward(d_out, cache):
    # TODO: scatter each d_out value to the cached argmax position in its window
    x_shape=cache['x_shape']
    argmax=cache['argmax']
    kernel=cache['kernel']
    stride=cache['stride']
    N,C,out_h,out_w=d_out.shape
    dx=np.zeros(x_shape,dtype=d_out.dtype)
    for n in range(N):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    h_start=i*stride
                    w_start=j*stride
                    idx=argmax[n,c,i,j]
                    r=idx//kernel
                    col=idx%kernel
                    dx[n,c,h_start+r,w_start+col]+=d_out[n,c,i,j]
    return dx

# Step 25 - relu_forward
def relu_forward(x):
    # TODO: Compute the elementwise ReLU and cache the input for backprop.
    out=np.maximum(0,x)
    cache={
        'x':x
    }
    return out,cache

# Step 26 - relu_backward
def relu_backward(d_out, cache):
    # TODO: mask the upstream gradient by the positive entries of the cached input.
    x=cache['x']
    return d_out*(x>0)

# Step 27 - flatten_forward
def flatten_forward(x):
    # TODO: reshape a 4D feature map into a 2D batch matrix and cache the original shape
    x_shape=x.shape
    N,C,H,W=x.shape
    out=x.reshape(N,-1)
    cache={
        'x_shape':x_shape
    }
    return out,cache

# Step 28 - flatten_backward
import numpy as np

def flatten_backward(d_out, cache):
    # TODO: reshape the upstream gradient back to the original 4D feature map shape.
    x_shape=cache['x_shape']
    N,C,H,W=x_shape
    dx=d_out.reshape(N,C,H,W)
    return dx

# Step 29 - linear_forward
def linear_forward(x, weights, bias):
    # TODO: compute X @ W + b and cache the inputs needed for backprop.
    out=x@weights+bias
    cache={
        'x':x,
        'weights':weights
    }
    return out,cache

# Step 30 - linear_grad_input
import numpy as np

def linear_grad_input(d_out, cache):
    """Gradient of a linear layer w.r.t. its input X."""
    # TODO: return dL/dX given d_out (N, D_out) and cache['weights'] (D_in, D_out)
    weights=cache['weights']
    N,D_out=d_out.shape
    dx=d_out@weights.T
    return dx

# Step 31 - linear_grad_weights
import numpy as np

def linear_grad_weights(x, dout):
    """Gradient of loss wrt linear-layer weights W of shape (D_in, D_out)."""
    # TODO: Compute the gradient of a linear layer's loss wrt its weight matrix W.
    dw=x.T@dout
    return dw

# Step 32 - linear_grad_bias
import numpy as np

def linear_grad_bias(dout):
    # TODO: Compute the bias gradient of a linear layer given upstream gradient dout.
    db=np.sum(dout,axis=0)
    return db

# Step 33 - linear_backward
def linear_backward(dout, cache):
    # TODO: combine input, weight, and bias gradients for a linear layer using the cache
    x=cache['x']
    dx=linear_grad_input(dout,cache)
    dw=linear_grad_weights(x,dout)
    db=linear_grad_bias(dout)
    return dx,dw,db

# Step 34 - softmax_cross_entropy_forward
def softmax_cross_entropy_forward(logits, y):
    # TODO: return the mean cross-entropy loss for logits (N, C) and integer labels y (N,).
    logits_softmax=stable_softmax(logits)
    p_log_mean=cross_entropy_loss(logits_softmax,y)
    return float(p_log_mean)+0.0

# Step 35 - softmax_cross_entropy_backward
def softmax_cross_entropy_backward(logits, y):
    # TODO: return the fused softmax-cross-entropy gradient of shape (N, C).
    logits_softmax=stable_softmax(logits)
    N,C=logits.shape
    y_onehot=one_hot(y,C)
    dlogits=(logits_softmax-y_onehot)/N
    return dlogits

# Step 36 - sgd_step
import numpy as np

def sgd_step(param, grad, lr):
    # TODO: return the SGD-updated parameter array (param - lr * grad).
    param_sgd=param-lr*grad
    return param_sgd

# Step 37 - adam_update_m
import numpy as np

def adam_update_m(m, grad, beta_one):
    # TODO: return the updated first moment estimate using beta_one and grad.
    m_new=beta_one*m+(1-beta_one)*grad
    return m_new

# Step 38 - adam_update_v
import numpy as np

def adam_update_v(v, grad, beta_two):
    # TODO: return the updated Adam second moment estimate as an EMA of squared gradients.
    v_new=beta_two*v+(1-beta_two)*(grad**2)
    return v_new

# Step 39 - adam_bias_correct
def adam_bias_correct(moment, beta, t):
    # TODO: return moment divided by (1 - beta**t) to undo Adam's zero-init bias.
    moment_new=moment/(1-beta**t)
    return moment_new

# Step 40 - adam_param_step
import numpy as np

def adam_param_step(param, m_hat, v_hat, lr, eps):
    # TODO: apply one Adam parameter update using bias-corrected moments
    param_new=param-lr*(m_hat)/(np.sqrt(v_hat)+eps)
    return param_new

# Step 41 - adam_step
import numpy as np

def adam_step(param, grad, m, v, t, lr, beta_one, beta_two, eps):
    # TODO: chain the four Adam helpers and return (new_param, new_m, new_v)
    new_m=adam_update_m(m,grad,beta_one)
    new_v=adam_update_v(v,grad,beta_two)
    m_hat=adam_bias_correct(new_m,beta_one,t)
    v_hat=adam_bias_correct(new_v,beta_two,t)
    new_p=adam_param_step(param,m_hat,v_hat,lr,eps)
    return new_p,new_m,new_v

# Step 42 - init_conv_layer
def init_conv_layer(out_channels, in_channels, kernel_size, seed=0):
    # TODO: Build He-initialized weights and a zero bias for a single conv layer.
    bias=init_zero_bias(out_channels)
    fan_in=in_channels*kernel_size*kernel_size
    weights_shape=(out_channels,in_channels,kernel_size,kernel_size)
    weights=he_init(weights_shape,fan_in,seed)
    dict={
        'W':weights,
        'b':bias
    }
    return dict

# Step 43 - init_linear_layer
def init_linear_layer(in_features, out_features, seed=0):
    # TODO: return {'W': He-init matrix (in_features, out_features), 'b': zero bias (out_features,)}
    bias=init_zero_bias(out_features)
    W_shape=in_features,out_features
    W=he_init(W_shape,in_features,seed)
    dict={
        'W':W,
        'b':bias
    }
    return dict

# Step 44 - init_lenet
def init_lenet(in_channels, num_classes, seed=0):
    # TODO: build conv1, conv2, fc1, fc2 with the right shapes and return them in a dict.
    conv1=init_conv_layer(6,in_channels,5)
    conv2=init_conv_layer(16,6,5)
    fc1=init_linear_layer(256,120)
    fc2=init_linear_layer(120,num_classes)
    dict={
        'conv1':conv1,
        'conv2':conv2,
        'fc1':fc1,
        'fc2':fc2
    }
    return dict

# Step 45 - forward_conv_block
def forward_conv_block(x, W, b, pool_size, stride, pad):
    # TODO: run conv2d -> relu -> maxpool2d and return (out, cache_dict)
    x_conv,conv_cache=conv2d_forward(x,W,b,stride,pad)
    x_relu_act,relu_cache=relu_forward(x_conv)
    x_pool,pool_cache=maxpool2d_forward(x_relu_act,pool_size,pool_size)
    dict={
        'conv_cache':conv_cache,
        'relu_cache':relu_cache,
        'pool_cache':pool_cache
    }
    return x_pool,dict

# Step 46 - forward_classifier_block
def forward_classifier_block(x, fc1, fc2):
    # TODO: run flatten -> linear -> relu -> linear and return logits plus a cache dict.
    x_flatten,flatten_cache=flatten_forward(x)
    linear1,fc1_cache=linear_forward(x_flatten,fc1['W'],fc1['b'])
    x_relu,relu_cache=relu_forward(linear1)
    linear2,fc2_cache=linear_forward(x_relu,fc2['W'],fc2['b'])
    cache={
        'fc1_cache':fc1_cache,
        'fc2_cache':fc2_cache,
        'flatten_cache':flatten_cache,
        'relu_cache':relu_cache
    }
    return linear2,cache

# Step 47 - lenet_forward
def lenet_forward(x, params):
    # TODO: run two conv blocks then the classifier block and return (logits, caches).
    w1=params['conv1']['W']
    b1=params['conv1']['b']
    w2=params['conv2']['W']
    b2=params['conv2']['b']

    fc1=params['fc1']
    fc2=params['fc2']
    block1,block1_cache=forward_conv_block(x,w1,b1,pool_size=2,stride=1,pad=0)
    block2,block2_cache=forward_conv_block(block1,w2,b2,pool_size=2,stride=1,pad=0)
    out,classifier=forward_classifier_block(block2,fc1,fc2)
    cache={
        'block1':block1_cache,
        'block2':block2_cache,
        'classifier':classifier
    }
    return out,cache

# Step 48 - backward_conv_block
def backward_conv_block(dout, cache):
    # TODO: backprop dout through the cached pool, relu, and conv layers in reverse order.
    conv_cache=cache['conv_cache']
    relu_cache=cache['relu_cache']
    pool_cache=cache['pool_cache']
    dx_pool=maxpool2d_backward(dout,pool_cache)
    dout_relu=relu_backward(dx_pool,relu_cache)
    dx,dW,db=conv2d_backward(dout_relu,conv_cache)
    return dx,dW,db

# Step 49 - backward_classifier_block
def backward_classifier_block(dlogits, cache):
    # TODO: backprop through fc2 -> relu -> fc1 -> flatten using the cached values
    fc1_cache=cache['fc1_cache']
    fc2_cache=cache['fc2_cache']
    flatten_cache=cache['flatten_cache']
    relu_cache=cache['relu_cache']
    dx,dw_2,db_2=linear_backward(dlogits,fc2_cache)
    dx=relu_backward(dx,relu_cache)
    dx,dw_1,db_1=linear_backward(dx,fc1_cache)
    dx=flatten_backward(dx,flatten_cache)
    grad={
        'dx':dx,
        'fc1':{
            'dW':dw_1,
            'db':db_1
        },
        'fc2':{
            'dW':dw_2,
            'db':db_2
        }
    }
    return grad

# Step 50 - lenet_backward
def lenet_backward(dlogits, caches):
    # TODO: walk classifier and conv block caches in reverse to assemble all gradients
    block1=caches['block1']
    block2=caches['block2']
    classifier=caches['classifier']
    grad=backward_classifier_block(dlogits,classifier)
    dx,dW2,db2=backward_conv_block(grad['dx'],block2)
    dx,dW1,db1=backward_conv_block(dx,block1)
    dict={
        'conv1':{
            'dW':dW1,'db':db1
        },
        'conv2':{
            'dW':dW2,'db':db2
        },
        'fc1':grad['fc1'],
        'fc2':grad['fc2']
    }
    return dict

# Step 51 - lenet_predict
def lenet_predict(x, params):
    # TODO: Return the argmax class index per sample from a LeNet forward pass.
    out,cache=lenet_forward(x,params)
    out=argmax_rows(out)
    return out

# Step 52 - build_synthetic_image_dataset
def build_synthetic_image_dataset(num_samples, num_classes, image_size, in_channels=1, seed=0):
    # TODO: Return (x, y) for a reproducible synthetic NCHW image dataset.
    rng=np.random.default_rng(seed)
    y=rng.integers(0,num_classes,size=num_samples)
    X=rng.standard_normal((num_samples,in_channels,image_size,image_size))
    class_means=np.linspace(-1,1,num_classes)
    X=X+class_means[y].reshape(num_samples,1,1,1)
    return X,y

# Step 53 - shuffle_indices
import numpy as np

def shuffle_indices(n, seed=0):
    # TODO: return a reproducible permutation of [0, n) as an int ndarray of shape (n,).
    np.random.seed(seed)
    return np.random.permutation(n)

# Step 54 - train_test_split
def train_test_split(x, y, test_fraction=0.2, seed=0):
    # TODO: partition x and y into train and test halves using a shared shuffled order.
    N=x.shape[0]
    n_test=int(N*test_fraction)
    idx=shuffle_indices(N,seed=0)
    x_test=x[idx[:n_test]]
    x_train=x[idx[n_test:]]
    y_test=y[idx[:n_test]]
    y_train=y[idx[n_test:]]
    return (x_train,y_train,x_test,y_test)

# Step 55 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 56 - train_step (not yet solved)
# TODO: implement

# Step 57 - train_one_epoch (not yet solved)
# TODO: implement

# Step 58 - train_loop (not yet solved)
# TODO: implement

# Step 59 - evaluate (not yet solved)
# TODO: implement

