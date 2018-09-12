def conv_bn_layer(input,ch_out,filter_size,stride,padding,active_type=paddle.activation.Relu(),ch_in=None):
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
    
def bn_conv_layer(input, ch_out, filter_size, stride, padding, active_type=paddle.activation.Relu(), ch_in=None):
    tmp = paddle.layer.batch_norm(input=input, act=active_type,  moving_average_fraction=0.999)
    return paddle.layer.img_conv(
        input=tmp,
        filter_size=filter_size,
        num_channels=ch_in,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=paddle.activation.Linear(),
        bias_attr=False)

def bottlenet(data, num_filter, stride, dim_match):
    conv1 = bn_conv_layer(data, int(num_filter * 0.25), 1,1,0)
    conv2 = bn_conv_layer(conv1, int(num_filter * 0.25), 3, stride, 1)
    conv3 = bn_conv_layer(conv2, int(num_filter), 1, 1, 0)
    if dim_match:
        shortcut = data
    else:
        shortcut = paddle.layer.img_conv(input=conv1, filter_size=1, num_filters=num_filter, stride=stride, padding=0,
            act=paddle.activation.Linear(), bias_attr=True)
    body = paddle.layer.addto(input=[conv3, shortcut], act=paddle.activation.Linear(), bias_attr=False)
    return body

def resnet(ipt, class_dim=7403):
    s = 30
    
    feature_maps = 128
    num_stages = 4
    units = [3, 6, 24, 3]
    filter_list = [64, 128, 256, 512, 1024]
    ipt_bn = ipt - 128.0

    body = conv_bn_layer(ipt_bn, ch_in=3, ch_out=filter_list[0], filter_size=7, stride=2, padding=3)
    body = conv_bn_layer(body, ch_out=filter_list[0], filter_size=3, stride=2, padding=1)
    
    for i in range(num_stages):
        if i == 0:
            conv1 = conv_bn_layer(body, ch_out=filter_list[0], filter_size=1, stride=1, padding=0)
            conv2 = conv_bn_layer(conv1, ch_out=filter_list[0], filter_size=3, stride=1, padding=1)
            conv3 = conv_bn_layer(conv2, ch_out=filter_list[0 + 1], filter_size=1, stride=1, padding=0)
            shortcut = conv_bn_layer(conv1, ch_out=filter_list[0 + 1], filter_size=1, stride=1, padding=0)
            body = paddle.layer.addto(input=[conv3, shortcut], act=paddle.activation.Linear(), bias_attr=False)
        else:
            body = bottlenet(body, filter_list[i + 1], 2, False)

        for j in range(units[i] - 1):
            body = bottlenet(body, filter_list[i + 1], 1, True)
    
    fea = paddle.layer.dropout(input=body, dropout_rate=0.5)
    fea = paddle.layer.fc(fea, size=feature_maps, act=paddle.activation.Linear(), bias_attr=False)
    fea = paddle.layer.batch_norm(input=fea, act=paddle.activation.Linear(), moving_average_fraction=0.999)
    
    o = paddle.layer.data(name='o',type=paddle.data_type.dense_vector(1))
    z = paddle.layer.data(name='z',type=paddle.data_type.dense_vector(class_dim))
    l = paddle.layer.data(name='l', type=paddle.data_type.dense_vector(class_dim))
    
    m = -0.0*l
    w = paddle.layer.fc(o, size=feature_maps * class_dim, name='w', act=paddle.activation.Linear(), bias_attr=False, 
    param_attr=paddle.attr.Param(name='w.w'))
    cos_sim = paddle.layer.cos_sim(fea, w, scale=1, size=class_dim)
    Am = paddle.layer.addto(input=[cos_sim, m], bias_attr=False) * s
    fc = paddle.layer.addto(input=[Am, z], act=paddle.activation.Softmax(), bias_attr=False)
    return fea, fc
