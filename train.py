#!/usr/bin/env python3

import os
import csv
import random
import signal

import cv2
import tensorflow as tf

import model
import config
from utils import *

def load_training_data():
    X = []
    Y = []
    with open(os.path.join(config.TRAINING_DATA_DIR, config.TRAINING_DATA_FILE), 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for center, left, right, steering, _, _, speed in reader:
            sa = float(steering)
            X.extend([center.strip(), left.strip(), right.strip()])
            Y.extend([sa, sa+config.ANGLE_DELTA_CORRECTION, sa-config.ANGLE_DELTA_CORRECTION])
            # X.extend([center.strip()])
            # Y.extend([sa])
    return X, Y
        
@static_vars(offset=0, orders=[])
def next_batch(X, Y):
    n_samples = len(X)
    if n_samples != len(Y):
        raise ValueError("Unmatched number of samples and label data")
    if n_samples < config.BATCH_SIZE:
        raise ValueError("Number of samples is less than batch size")
    if n_samples != len(next_batch.orders):
        next_batch.orders = list(range(n_samples)) 
    if next_batch.offset + config.BATCH_SIZE > n_samples:
        next_batch.offset = 0
        random.shuffle(next_batch.orders)

    X_batch = []
    Y_batch = []
    for _ in range(config.BATCH_SIZE):
        im = cv2.imread(os.path.join(config.TRAINING_DATA_DIR, X[next_batch.offset]), cv2.IMREAD_COLOR)

        if next_batch.offset % 2 == 0:
            X_batch.append(model.preprocess(cv2.flip(im, 1)))
            Y_batch.append(-Y[next_batch.offset])
        else:
            X_batch.append(model.preprocess(im))
            Y_batch.append(Y[next_batch.offset])

        next_batch.offset += 1
    
    return X_batch, Y_batch

X, keep_prob, pred = model.build_net()
Y = tf.placeholder(tf.float32, [None], "human_data")

loss = model.build_loss(Y, pred, config.BETA, tf.trainable_variables())

global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
optimizer = tf.train.AdamOptimizer(1e-6, name="optimizer").minimize(loss, global_step=global_step)

samples, labels = load_training_data()
tf.summary.scalar("loss", loss)
summary_op = tf.summary.merge_all()

exit_request = [False]
def exit_handler(*args):
    exit_request[0] = True
signal.signal(signal.SIGINT, exit_handler)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    cp = tf.train.get_checkpoint_state(config.SAVE_FOLDER)
    if cp and cp.model_checkpoint_path:
        saver.restore(sess, cp.model_checkpoint_path)
        print("Checkpoint Loaded", cp.model_checkpoint_path)
    writer = tf.summary.FileWriter(config.LOG_FOLDER, sess.graph)

    while True:
        X_batch, Y_batch = next_batch(samples, labels)

        _, step, loss_batch, summary = sess.run([optimizer, global_step, loss, summary_op],
            feed_dict={ X: X_batch,
                        Y: Y_batch,
                        keep_prob: config.KEEP_PROB})

        if step % config.LOG_INTERVAL == 0:
            print("Step: {}, Loss: {}".format(step, loss_batch))
            writer.add_summary(summary, step)
        
        if exit_request[0] or step % config.SAVE_INTERVAL == 0:
            print("Model Saved")
            saver.save(sess, config.SAVE_FOLDER, step)
            if exit_request[0]:
                exit(0)

