'''
BASED ON:
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''

import os
import cv2

import numpy as np

import onnxruntime


import multiprocessing

def init_session(model_path):
    EP_list = ['CPUExecutionProvider']
    so = onnxruntime.SessionOptions()
    so.log_severity_level = 3
    sess = onnxruntime.InferenceSession(model_path, so, providers=EP_list)
    
    return sess

class PickableInferenceSession: # This is a wrapper to make the current InferenceSession class pickable.
    #https://github.com/microsoft/onnxruntime/issues/7846
    def __init__(self, model_path):
        self.model_path = model_path
        self.sess = init_session(self.model_path)

    def run(self, *args):
        return self.sess.run(*args)

    def __getstate__(self):
        return {'model_path': self.model_path}

    def __setstate__(self, values):
        self.model_path = values['model_path']
        self.sess = init_session(self.model_path)
        
    def get_inputs(self):
        return self.sess.get_inputs()


class FaceGAN(object):
    def __init__(self, size=512, model=None):

        base = os.path.dirname(os.path.realpath(__file__))
        self.onnxfile = os.path.join(base, '.', model+'.onnx')

        self.resolution = size
        self.ort_session = PickableInferenceSession(self.onnxfile)
        #self.load_model(channel_multiplier, narrow)
        

    def load_model(self):
    
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        # Use OpenMP optimizations. Only useful for CPU, has little impact for GPUs.
        #print(onnxruntime.get_device())
        #so.intra_op_num_threads = min(1, multiprocessing.cpu_count()-2)
        self.ort_session = onnxruntime.InferenceSession(self.onnxfile, so)
            
   
    def process(self, img, output_size=None, is_tanh=True, preserve_size=False):
        img = cv2.resize(img, (self.resolution, self.resolution))
                
        img = img - 0.5
        img = img / 0.5 
        img = np.transpose(img, (2,0,1))
        img = np.expand_dims(img, 0)
        img = np.flip(img, 1).copy()
        
        ort_inputs = {self.ort_session.get_inputs()[0].name: img }
        ort_outs = self.ort_session.run(None, ort_inputs)
        del img

        out = ort_outs[0]
        out = out * 0.5 
        out = out + 0.5
        out = np.squeeze(out, 0)
        out = np.transpose(out, (1,2,0))
        out = np.flip(out, 2).copy()
        out = np.clip(out, 0., 1.)
        out = out.astype("float32")
        
        return out
        

