#!/usr/bin/env python3

import tensorflow as tf

from drive import Simulator
import model
import config
from utils import *

if __name__ == "__main__":

    X, keep_prob, pred = model.build_net(trainable=False)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    cp = tf.train.get_checkpoint_state(config.SAVE_FOLDER)
    if cp and cp.model_checkpoint_path:
        saver.restore(sess, cp.model_checkpoint_path)
        print("Checkpoint Loaded", cp.model_checkpoint_path)

    def controller(data):
        steering = sess.run(pred,
            feed_dict = { X: [model.preprocess(base64_to_cvmat(data["image"]))],
                          keep_prob: 1.0})
        steering = steering[0]
        return steering, 1.0

    sim = Simulator(controller)
    sim.init()
    sim.run()
