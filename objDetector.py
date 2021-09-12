#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: prasanth
"""

from flask import Flask, Response, render_template
from detection import Video

app=Flask(__name__)

def detection_feed_generator(feed):
    '''
    Generates detections on a video feed
    '''
    while True:
        frame=feed.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type:  image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



@app.route('/')
def index():
    '''
    Default route for the app
    '''
    return render_template('index.html')

@app.route('/video')
def video():
    '''
    Route for the video stream
    '''
    return Response(detection_feed_generator(Video()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(host='0.0.0.0',port=8000)