#!/usr/bin/env python3

import socketio
import eventlet
import eventlet.wsgi
from flask import Flask

class PIController:
    def __init__(self, Kp=0.1, Ki=0.002):
        self.Kp = Kp
        self.Ki = Ki
        self.target = 22 
        self.integral = 0.0

    def init(self):
        self.integral = 0.0

    def set_target(self, sp):
        self.target = sp

    def update(self, measurement):
        err = self.target - measurement
        self.integral += err
        return self.Kp*err + self.Ki*self.integral


class Simulator:
    def __init__(self, call_back_fn):
        self.sio = socketio.Server()
        self.app = Flask(__name__)
        self.call_back_fn = call_back_fn

    def init(self):
        @self.sio.on("telemetry")
        def telemetry(sid, data):
            # steering_angle = data["steering_angle"]
            # throttle = data["throttle"]
            # speed = data["speed"]
            # im_string = data["image"]
            steering_angle, throttle = self.call_back_fn(data)
            print("Steering: {}, Throttle: {}".format(steering_angle, throttle))
            self.send(steering_angle, throttle)

        @self.sio.on("connect")
        def connect(sid, environ):
            print("connected", sid)
            self.send(0, 0)

    def run(self):
        eventlet.wsgi.server(eventlet.listen(('', 4567)), socketio.Middleware(self.sio, self.app))

    def send(self, steering_angle, throttle):
        self.sio.emit("steer",
             data={ "steering_angle": str(steering_angle),
                    "throttle": str(throttle) },
             skip_sid=True)

if __name__ == "__main__":
    def controller(data):
        return 0.0, 1.0

    sim = Simulator(controller)
    sim.init()
    sim.run()
