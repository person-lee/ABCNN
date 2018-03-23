import tensorflow as tf
import numpy as np

#----------------------------- cal attention -------------------------------
#input_q, input_a (batch_size, rnn_size, seq_len)
def cal_attention(input_q, input_a, U):
    batch_size = int(input_q.get_shape()[0])
    U = tf.tile(tf.expand_dims(U, 0), [batch_size, 1, 1])
    G = tf.batch_matmul(tf.batch_matmul(input_q, U, True), input_a)
    delta_q = tf.nn.softmax(tf.reduce_max(G, 1), 1)
    delta_a = tf.nn.softmax(tf.reduce_max(G, 2), 1)

    return delta_q, delta_a


def cos_sim(feat_q, feat_a):
    norm_q = tf.sqrt(tf.reduce_sum(tf.mul(feat_q, feat_q), 1))
    norm_a = tf.sqrt(tf.reduce_sum(tf.mul(feat_a, feat_a), 1))
    mul_q_a = tf.reduce_sum(tf.mul(feat_q, feat_a), 1)
    cos_sim_q_a = tf.div(mul_q_a, tf.mul(norm_q, norm_a))
    return cos_sim_q_a

def np_cos_sim(feat_q, feat_a):
    norm_q = np.sqrt(np.sum(np.multiply(feat_q, feat_q)))
    norm_a = np.sqrt(np.sum(np.multiply(feat_a, feat_a)))
    mul_q_a = np.sum(np.multiply(feat_q, feat_a))
    cos_sim_q_a = 1. * mul_q_a / np.multiply(norm_q, norm_a)
    return cos_sim_q_a

# return 1 output of lstm cells after pooling, lstm_out(batch, step, rnn_size * 2)
def max_pooling(lstm_out):
    height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])       # (step, length of input for one step)
    
    # do max-pooling to change the (sequence_lenght) tensor to 1-lenght tensor
    lstm_out = tf.expand_dims(lstm_out, -1)
    output = tf.nn.max_pool(
    	lstm_out,
    	ksize=[1, height, 1, 1],
    	strides=[1, 1, 1, 1],
    	padding='VALID')
    
    output = tf.reshape(output, [-1, width])
    
    return output

def avg_pooling(lstm_out):
	height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])       # (step, length of input for one step)

	# do max-pooling to change the (sequence_lenght) tensor to 1-lenght tensor
	lstm_out = tf.expand_dims(lstm_out, -1)
	output = tf.nn.avg_pool(
		lstm_out,
		ksize=[1, height, 1, 1],
		strides=[1, 1, 1, 1],
		padding='VALID')

	output = tf.reshape(output, [-1, width])

	return output

def cal_loss_and_acc(ori_cand, ori_neg):
    # the target function 
    zero = tf.fill(tf.shape(ori_cand), 0.0)
    margin = tf.fill(tf.shape(ori_cand), 0.15)
    with tf.name_scope("loss"):
        losses = tf.maximum(zero, tf.sub(margin, tf.sub(ori_cand, ori_neg)))
        loss = tf.reduce_sum(losses) 
    # cal accurancy
    with tf.name_scope("acc"):
        correct = tf.equal(zero, losses)
        acc = tf.reduce_mean(tf.cast(correct, "float"), name="acc")
    return loss, acc

def ortho_weight(ndim):
    W = tf.random_normal([ndim, ndim], stddev=0.1)
    s, u, v = tf.svd(W)
    return u

def uniform_weight(nin, nout=None, scale=0.05):
    if nout == None:
        nout = nin
    W = tf.random_uniform(shape=[nin, nout], minval=-scale, maxval=-scale)
    return W


def make_attention_mat(x1, x2):
    # x1, x2 = [batch, height, width, 1] = [batch, d, s, 1]
    # x2 => [batch, height, 1, width]
    # [batch, width, wdith] = [batch, s, s]
    euclidean = tf.sqrt(tf.reduce_sum(tf.square(x1 - tf.matrix_transpose(x2)), 1))
    return 1 / (1 + euclidean)

def w_pool(variable_scope, x, attention, filter_size, seq_len, model_type="ABCNN3"):
    # x: [batch, di, s+w-1, 1]
    # attention: [batch, s+w-1]
    with tf.variable_scope(variable_scope + "-w_pool"):
        if model_type == "ABCNN2" or model_type == "ABCNN3":
            pools = []
            # [batch, s+w-1] => [batch, 1, s+w-1, 1]
            attention = tf.transpose(tf.expand_dims(tf.expand_dims(attention, -1), -1), [0, 2, 1, 3])

            for i in range(seq_len):
                # [batch, di, w, 1], [batch, 1, w, 1] => [batch, di, 1, 1]
                pools.append(tf.reduce_sum(x[:, :, i:i + filter_size, :] * attention[:, :, i:i + filter_size, :],
                                           2,
                                           keep_dims=True))

            # [batch, num_filters, seq_len, 1]
            w_ap = tf.concat(2, pools, name="w_ap")
        else:
	    #w_ap = tf.nn.max_pool(
	    #	x,
	    #	ksize=[1, 1, pool_width, 1],
	    #	strides=[1, 1, 1, 1],
	    #	padding='VALID', name="w_ap")
            w_ap = tf.nn.avg_pool(
		x,
		ksize=[1, 1, filter_size, 1],
		strides=[1, 1, 1, 1],
		padding='VALID', name="w_ap")
            # [batch, di, s, 1]

        return w_ap

def all_pool(variable_scope, x, seq_len, filter_size, num_filters, vec_len):
    with tf.variable_scope(variable_scope + "-all_pool"):
        if variable_scope.startswith("input"):
            pool_width = seq_len
            d = vec_len
        else:
            pool_width = seq_len + filter_size - 1
            d = num_filters

	#all_ap = tf.nn.max_pool(
	#	x,
	#	ksize=[1, 1, pool_width, 1],
	#	strides=[1, 1, 1, 1],
	#	padding='VALID', name="all_ap")
	all_ap = tf.nn.avg_pool(
		x,
		ksize=[1, pool_width, 1, 1],
		strides=[1, 1, 1, 1],
		padding='VALID', name="all_ap")
        # [batch, di, 1, 1]

        # [batch, di]
        all_ap_reshaped = tf.reshape(all_ap, [-1, d])
        #all_ap_reshaped = tf.squeeze(all_ap, [2, 3])

        return all_ap_reshaped

def pad_for_wide_conv(x, w):
    return tf.pad(x, np.array([[0, 0], [0, 0], [w - 1, w - 1], [0, 0]]), "CONSTANT", name="pad_wide_conv")

def build_path(prefix, data_type, model_type, num_layers, postpix=""):
    return prefix + data_type + "-" + model_type + "-" + str(num_layers) + postpix

def getcomlen(firststr, secondstr):
    comlen = 0
    while firststr and secondstr:
        if firststr[0] == secondstr[0]:
            comlen += 1
            firststr = firststr[1:]
            secondstr = secondstr[1:]
        else:
            break
    return comlen

def lcs_continuous(input_x, input_y):
    """
    the length of common sequence(need continuous)
    """
    max_common_len = 0
    common_index = 0
    for xtemp in range(0, len(input_x)):
        for ytemp in range(0, len(input_y)):
            com_temp = getcomlen(input_x[xtemp: len(input_x)], input_y[ytemp: len(input_y)])
            if com_temp > max_common_len:
                max_common_len = com_temp
                common_index = xtemp
    return max_common_len

def lcs(input_x, input_y):
    """
    the length of common sequence(do not need continuous)
    """
    commList = set()
    lena = len(input_x)  
    lenb = len(input_y)  
    c = [[0 for i in range(lenb+1)] for j in range(lena+1)]  
    for i in range(lena):  
        for j in range(lenb):  
            if input_x[i] == input_y[j]:  
                c[i + 1][j + 1] = c[i][j] + 1  
                commList.add(input_x[i])
            elif c[i + 1][j] > c[i][j + 1]:  
                c[i + 1][j + 1] = c[i + 1][j]  
            else:  
                c[i + 1][j + 1] = c[i][j + 1]  
    return c[lena][lenb], commList

if __name__ == "__main__":
    print lcs_continous("1112223", "1223331")
    print lcs("1112223", "1223331")


