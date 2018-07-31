
import numpy as np
import paddle.v2 as paddle


def conv_bn_layer(input,
                  ch_out,
                  filter_size,
                  stride,
                  padding,
                  active_type=paddle.activation.Relu(),
                  ch_in=None):
    tmp = paddle.layer.img_conv(
        input=input,
        filter_size=filter_size,
        num_channels=ch_in,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=paddle.activation.Linear(),
        bias_attr=False)
    return paddle.layer.batch_norm(input=tmp, act=active_type, moving_average_fraction=0.999)


def shortcut(ipt, ch_in, ch_out, stride):
    if ch_in != ch_out:
        return conv_bn_layer(ipt, ch_out, 1, stride, 0, paddle.activation.Linear())
    else:
        return ipt

def basicblock(ipt, ch_in, ch_out, stride):
    tmp = conv_bn_layer(ipt, ch_out, 3, stride, 1)
    tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, paddle.activation.Linear())
    short = shortcut(ipt, ch_in, ch_out, stride)
    return paddle.layer.addto(input=[tmp, short], act=paddle.activation.Relu())


def layer_warp(block_func, ipt, ch_in, ch_out, count, stride):
    tmp = block_func(ipt, ch_in, ch_out, stride)
    for i in range(1, count):
        tmp = block_func(tmp, ch_out, ch_out, 1)
    return tmp


def resnet_baseline(ipt, class_dim=1036):
    # resnet
    n = 1
    feature_maps = 128
    ipt_bn = ipt - 128.0
    conv1 = conv_bn_layer(ipt_bn, ch_in=1, ch_out=4, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 4, 8, n, 1)
    res2 = layer_warp(basicblock, res1, 8, 16, n, 2)
    res3 = layer_warp(basicblock, res2, 16, 32, n, 2)
    res4 = layer_warp(basicblock, res3, 32, 64, n, 2)
    res5 = layer_warp(basicblock, res4, 64, feature_maps, n, 2)

    pool = paddle.layer.img_pool(input=res5, name='pool', pool_size=4, stride=1, pool_type=paddle.pooling.Avg())
    fc = paddle.layer.fc(input=pool, size=class_dim, act=paddle.activation.Softmax())
    return pool, fc






