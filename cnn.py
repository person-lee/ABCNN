# --*-- coding:utf-8 --*--

import tensorflow as tf
from utils import pad_for_wide_conv, w_pool, all_pool, make_attention_mat

def CNN_layer(variable_scope, x1, x2, seq_len, vec_len, num_filters, filter_size, l2_reg, model_type):
    # x1, x2 = [batch, d, s, 1]
    with tf.variable_scope(variable_scope):
        if model_type == "ABCNN1" or model_type == "ABCNN3":
            with tf.name_scope("att_mat"):
                aW = tf.get_variable(name="aW",
                                     shape=(seq_len, vec_len),
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg))

                # [batch, seq_len, seq_len]
                att_mat = make_attention_mat(x1, x2)

                # [batch, s, s] * [s,d] => [batch, s, d]
                # matrix transpose => [batch, d, s]
                # expand dims => [batch, vec_len, seq_len, 1]
                x1_a = tf.expand_dims(tf.transpose(tf.einsum("ijk,kl->ijl", att_mat, aW), [0, 2, 1]), -1)
                x2_a = tf.expand_dims(tf.transpose(
                    tf.einsum("ijk,kl->ijl", tf.transpose(att_mat, [0, 2, 1]), aW), [0, 2, 1]), -1)

                # [batch, vec_len, seq_len, 2]
                x1 = tf.transpose(tf.concat(3, [x1, x1_a]), [0, 2, 1, 3])
                x2 = tf.transpose(tf.concat(3, [x2, x2_a]), [0, 2, 1, 3])

        # shape [batch_size, num_filters, seq_len + filter_size - 1, 1]
        #left_conv = convolution(name_scope="left", x=pad_for_wide_conv(x1, filter_size), vec_len=vec_len, num_filters=num_filters, filter_size=filter_size, l2_reg=l2_reg, reuse=False)
        #right_conv = convolution(name_scope="right", x=pad_for_wide_conv(x2, filter_size), vec_len=vec_len, num_filters=num_filters, filter_size=filter_size, l2_reg=l2_reg, reuse=True)
        left_conv = CNN(x1, seq_len, vec_len, filter_size, num_filters)
        right_conv = CNN(x2, seq_len, vec_len, filter_size, num_filters)
        return left_conv, right_conv

        #left_attention, right_attention = None, None

        #if model_type == "ABCNN2" or model_type == "ABCNN3":
        #    # [batch, seq_len+filter_size-1, seq_len+filter_size-1]
        #    att_mat = make_attention_mat(left_conv, right_conv)
        #    # [batch, seq_len+filter_size-1], [batch, seq_len+filter_size-1]
        #    left_attention, right_attention = tf.reduce_sum(att_mat, 2), tf.reduce_sum(att_mat, 1)

        ##shape [batch_size, num_filters, seq_len, 1]
        #left_wp = w_pool(variable_scope="left", x=left_conv, attention=left_attention, filter_size=filter_size, seq_len=seq_len, model_type=model_type)
        ##shape [batch_size, num_filters]
        #left_ap = all_pool("left", left_conv, seq_len, filter_size, num_filters, vec_len)
        #right_wp = w_pool(variable_scope="right", x=right_conv, attention=right_attention, filter_size=filter_size, seq_len=seq_len, model_type=model_type)
        #right_ap = all_pool("right", right_conv, seq_len, filter_size, num_filters, vec_len)

        #return left_wp, left_ap, right_wp, right_ap

def convolution(name_scope, x, vec_len, num_filters, filter_size, l2_reg, reuse):
    with tf.name_scope(name_scope + "-conv"):
        with tf.variable_scope("conv") as scope:
            conv = tf.contrib.layers.conv2d(
                inputs=x,
                num_outputs=num_filters,
                kernel_size=(vec_len, filter_size),
                stride=1,
                padding="VALID",
                activation_fn=tf.nn.tanh,
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                biases_initializer=tf.constant_initializer(1e-04),
                reuse=reuse,
                trainable=True,
                scope=scope
            )
            # Weight: [filter_height, filter_width, in_channels, out_channels]
            # output: [batch, 1, input_width+filter_Width-1, out_channels] == [batch, 1, s+w-1, di]

            # [batch, num_filters, seq_len+filter_size-1, 1]
            conv_trans = tf.transpose(conv, [0, 3, 2, 1], name="conv_trans")
            return conv_trans

def CNN(input_x, sequence_len, embedding_size, filter_sizes, num_filters):
    pooled_outputs = []
    for idx, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s"% (filter_size)):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            filter_weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="filter_weight")
            filter_bias = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="filter_bias")
            
            # convolution (batch_size, sequence_len - filter_size + 1, in_channels, out_channnels)
            conv = tf.nn.conv2d(input_x, filter_weight, strides=[1,1,1,1], padding="VALID")

            # apply nonlinearity
            relu_output = tf.nn.relu(tf.nn.bias_add(conv, filter_bias))

            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                relu_output,
                ksize=[1, sequence_len - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID')
            pooled_outputs.append(pooled)
    cnn_output = tf.squeeze(tf.concat(3, pooled_outputs), [1, 2])
    return cnn_output
