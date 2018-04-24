import base64
import cv2
import numpy as np

def static_vars(**args):
    def decorate(fn):
        for var in args:
            setattr(fn, var, args[var])
        return fn
    return decorate


def base64_to_cvmat(base64_str):
    return cv2.imdecode(np.frombuffer(base64.b64decode(base64_str), np.uint8), cv2.IMREAD_COLOR)
