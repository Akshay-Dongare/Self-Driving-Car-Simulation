################################################################
#to remove warning msgs
print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
################################################################

import tensorflow as tf
import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("Model_Path",help="Specify the path to the ML model you wish to test")
args = parser.parse_args()

 
#### FOR REAL TIME COMMUNICATION BETWEEN CLIENT AND SERVER
sio = socketio.Server()
#### FLASK IS A MICRO WEB FRAMEWORK WRITTEN IN PYTHON
app = Flask(__name__)  # '__main__'
 
maxSpeed = 10

 
def preProcess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img
 
 
@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = preProcess(image)
    image = np.array([image])
    steering = float(model.predict(image))
    throttle = 1.0 - speed / maxSpeed
    print(f'{steering}, {throttle}, {speed}')
    sendControl(steering, throttle)
 
 
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    sendControl(0, 0)
 
 
def sendControl(steering, throttle):
    sio.emit('steer', data={
        'steering_angle': steering.__str__(),
        'throttle': throttle.__str__()
    })
 
 
if __name__ == '__main__':
    #model = load_model('My Drivers/earlystop.h5')
    
    print("Loading model from path",args.Model_Path, "...")
    model = load_model(args.Model_Path)
    #keras_model1 = load_model('../My Drivers/earlystop.h5')
    #keras_model2 = load_model('../My Drivers/leftist.h5')
    #models = [keras_model1, keras_model2]
    #model_input = tf.keras.Input(shape=(125, 125, 3))
    #model_outputs = [model(model_input) for model in models]
    #ensemble_output = tf.keras.layers.Average()(model_outputs)
    #ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_output)
    app = socketio.Middleware(sio, app)
    ### LISTEN TO PORT 4567
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)