import os
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import sys
import datetime
from data_inference import data_inference

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(threshold=np.inf)

def get_stn_index(station):
    if station == 'EW24' or station == 'NS1':
        station = 'NS1&EW24'
    if station == 'EW13' or station == 'NS25':
        station = 'NS25&EW13'
    if station == 'EW14' or station == 'NS26':
        station = 'NS26&EW14'

    if station == 'NS1&EW24':
        return 0
    if station == 'NS25&EW13':
        return 1
    if station == 'NS26&EW14':
        return 2

    ret = -1
    line = station[:2]
    num = int(station[2:])
    if line == 'EW':
        if num in range(1,13):
            ret = num + 2
        if num in range(15,24):
            ret = num
        if num in range(25,30):
            ret = num - 1
    else:
        if num in range(2,6):
            ret = num + 27
        if num in range(7,12):
            ret = num + 26
        if num in range(13,25):
            ret = num + 25
        if num in range(27,29):
            ret = num + 23

    return ret


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    Weights = tf.Variable(np.random.rand(in_size,out_size),dtype=tf.float32)/np.sqrt(in_size/2)
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    outputs = tf.layers.batch_normalization(Wx_plus_b, training=True)
    if activation_function is None:
        outputs = outputs
    else:
        outputs = activation_function(outputs)

    return outputs


station_list = ['NS1&EW24', 'NS25&EW13', 'NS26&EW14'] + ['EW'+str(q) for q in range(1,13)] \
               + ['EW'+str(q) for q in range(15,24)] + ['EW'+str(q) for q in range(25,30)] \
               + ['NS'+str(q) for q in range(2,6)] + ['NS'+str(q) for q in range(7,12)] \
                + ['NS'+str(q) for q in range(13,25)] + ['NS'+str(q) for q in range(27,29)]

if __name__ == '__main__':
    argv = sys.argv
    hour = int(argv[1])
    group_num = int(argv[2])
    # start_date = argv[3]  # Format: '2016-01-01', '1/1/2016' or '20160101'
    # end_date = argv[4]  # Format: '2016-01-31', '1/31/2016' or '20160131'
    # dates_ori = pd.bdate_range(start_date, end_date)
    # n_epochs = len(dates_ori)
    date = argv[3]
    dates_ori = pd.bdate_range(end=date,periods=60)
    n_epochs = len(dates_ori)-1
    dates = [datetime.datetime.strftime(dates_ori[i], '%Y-%m-%d') for i in range(n_epochs)]

    stn_division = []
    if group_num == 1:
        stn_division = ['EW1', 'EW8', 'EW15', 'EW22']
    if group_num == 2:
        stn_division = ['EW2', 'EW9', 'EW16', 'EW23']
    if group_num == 3:
        stn_division = ['EW3', 'EW10', 'EW17', 'NS1&EW24']
    if group_num == 4:
        stn_division = ['EW4', 'EW11', 'EW18', 'EW25']
    if group_num == 5:
        stn_division = ['EW5', 'EW12', 'EW19', 'EW26']
    if group_num == 6:
        stn_division = ['EW6', 'NS25&EW13', 'EW20', 'EW27']
    if group_num == 7:
        stn_division = ['EW7', 'NS26&EW14', 'EW21', 'EW28']
    if group_num == 8:
        stn_division = ['NS2', 'NS9', 'NS16', 'NS22']
    if group_num == 9:
        stn_division = ['NS3', 'NS10', 'NS17', 'NS23']
    if group_num == 10:
        stn_division = ['NS4', 'NS11', 'NS18', 'NS24']
    if group_num == 11:
        stn_division = ['NS5', 'NS13', 'NS19', 'NS27']
    if group_num == 12:
        stn_division = ['NS7', 'NS14', 'NS20', 'NS28']
    if group_num == 13:
        stn_division = ['NS8', 'NS15', 'NS21', 'EW29']

    # batch_size = 512
    hidden_layers = 2
    hidden1_units = 120
    hidden2_units = 80
    n_input = 192
    n_classes = 52
    learning_rate_base = 0.1
    learning_rate_decay = 0.95
    decay_steps = n_epochs
    training_times = 100

    xs = tf.placeholder(tf.float32, [None, n_input], name="xs")
    ys = tf.placeholder(tf.float32, [None, n_classes], name="ys")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    l1 = add_layer(xs, n_input, hidden1_units, 'hidden_layer_1', activation_function=tf.nn.relu)
    l2 = add_layer(l1, hidden1_units, hidden2_units, 'hidden_layer_2', activation_function=tf.nn.relu)
    l3 = add_layer(l2, hidden2_units, n_classes, 'prediction_layer', activation_function=tf.nn.softmax)
    prediction = tf.multiply(l3, tf.ones([1, n_classes]), name="prediction")

    loss = tf.reduce_mean(-ys * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)), name="loss")
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, decay_steps, learning_rate_decay)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        st = time.time()
        sess.run(init)
        stn_division_index = [get_stn_index(stn_division[i]) for i in range(len(stn_division))]

        for m in range(training_times):
            for epoch in range(n_epochs):
                for batch in stn_division_index:
                    batch_xs, batch_ys = data_inference(dates[epoch], station_list[batch], hour)
                    sess.run(train_op, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob:1.0})

        end = time.time()
        print('Training is finished! Time cost:', int(end-st), 'seconds')
        save_path = saver.save(sess, "model_trans_prob_" + str(group_num))
        print('Model saved in path:', save_path)