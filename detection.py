#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: prasanth
"""
import numpy as np
import cv2, torch
import torchvision.transforms as transforms
from yolov3.models import load_model
from yolov3.utils.utils import load_classes, non_max_suppression, rescale_boxes
from yolov3.utils.transforms import Resize, DEFAULT_TRANSFORMS


class Video(object):
    '''
    Convinience object for video capture and detection
    '''
    
    def __init__(self):
        # Capture the video
        self.video=cv2.VideoCapture(-1) 
        # Create YOLOv3 model pre-trained on COCO dataset and load weights
        self.yolo = load_model('yolov3/yolov3.cfg', 
                               'yolov3/weights/yolov3.weights') 
        self.yolo.eval()
        # Load the COCO dataset classes
        self.classes = load_classes('yolov3/coco.names')
        
    def __del__(self):
        self.video.release()
        
    def get_frame(self):
        '''
        Perform inference and draw bounding boxes for each frame
        '''
        ret,frame=self.video.read()
        if not ret :
            ret,jpg=cv2.imencode('.jpg',np.zeros([256,256,3]))
            return jpg.tobytes()
        else:
            detections = self.detect_on_frame(np.array(frame))
            for detection in detections:
                x,y,x2,y2,conf,clas = detection
                x,y,x2,y2 = np.array([x,y,x2,y2]).astype(int)
                coco_class = self.classes[int(clas)]
                cv2.rectangle(frame, (x,y), (x2,y2), (255,0,255), 1)
                cv2.putText(frame, 
                            '{0:} {1:.2f}'.format(coco_class, conf),
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            ret,jpg=cv2.imencode('.jpg',frame)
            return jpg.tobytes()
        
    def detect_on_frame(self, frame, img_size=416, conf_thres=0.5, nms_thres=0.5):
        """Inferences one frame with model.

        :param frame: Frame to inference
        :type frame: nd.array
        :param img_size: Size of each image dimension for yolo, defaults to 416
        :type img_size: int, optional
        :param conf_thres: Object confidence threshold, defaults to 0.5
        :type conf_thres: float, optional
        :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
        :type nms_thres: float, optional
        :return: Detections on image with each detection in the format: [x1, y1, x2, y2, confidence, class]
        :rtype: nd.array
        """
        
        # Configure input
        input_img = transforms.Compose([
            DEFAULT_TRANSFORMS,
            Resize(img_size)])(
                (frame, np.zeros((1, 5))))[0].unsqueeze(0)
    
        if torch.cuda.is_available():
            input_img = input_img.to("cuda")
    
        # Get detections
        with torch.no_grad():
            detections = self.yolo(input_img)
            detections = non_max_suppression(detections, conf_thres, nms_thres)
            detections = rescale_boxes(detections[0], img_size, frame.shape[:2])
        return detections.numpy()