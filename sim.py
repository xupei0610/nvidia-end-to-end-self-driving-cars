#!/usr/bin/env python3

import tensorflow as tf

from drive import Simulator, PIController
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

    @static_vars(speed_controller=PIController())
    def controller(data):
        steering = sess.run(pred,
            feed_dict = { X: [model.preprocess(base64_to_cvmat(data["image"]), config.INPUT_IMAGE_CROP)],
                          keep_prob: 1.0})[0]
        throttle = controller.speed_controller.update(float(data["speed"]))
        return steering, throttle

    sim = Simulator(controller)
    sim.init()
    sim.run()
