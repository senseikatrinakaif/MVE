import multiprocessing
import operator

import numpy as np

from core import mathlib
from core.interact import interact as io
from core.leras import nn
from facelib import FaceType
from models import ModelBase
from samplelib import *

from pathlib import Path

from utils.label_face import label_face_filename

import onnxruntime

def init_session(model_path):
    EP_list = ['CPUExecutionProvider'] #TODO 
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

class DFMModel(ModelBase):


    #override
    def on_initialize(self):
        
        self.model_data_format =  "NHWC"
        nn.initialize(data_format=self.model_data_format)
        self._model_type = -1
        self._model_path = model_path = self.get_strpath_storage_for_dfm_file()

        sess = self._sess = PickableInferenceSession(model_path)

        inputs = sess.get_inputs()

        if len(inputs) == 0:
            raise Exception(f'Invalid model {model_path}')
        else:
            if 'in_face' not in inputs[0].name:
                raise Exception(f'Invalid model {model_path}')
            else:
                self._input_height, self._input_width = inputs[0].shape[1:3]
                self._model_type = 1
                if len(inputs) == 2:
                    if 'morph_value' not in inputs[1].name:
                        raise Exception(f'Invalid model {model_path}')
                    self._model_type = 2
                elif len(inputs) > 2:
                    raise Exception(f'Invalid model {model_path}')

        self.face_type = {'h'  : FaceType.HALF,
                          'mf' : FaceType.MID_FULL,
                          'f'  : FaceType.FULL,
                          'wf' : FaceType.WHOLE_FACE,
                          'custom' : FaceType.CUSTOM,
                          'head' : FaceType.HEAD}[ self.face_type ]
    

    def get_strpath_storage_for_dfm_file(self):
        return str( self.saved_models_path / ( self.get_model_name() + '.dfm') )
    

    def predictor_func (self, face, morph_value=1.0):

        img = nn.to_data_format(face[None,...], self.model_data_format, "NHWC")
        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            img /= 255.0

        if self._model_type == 1:
           mask_src_dstm, bgr, mask_dst_dstm = self._sess.run(None, {'in_face:0': img})
        elif self._model_type == 2:
            mask_src_dstm, bgr, mask_dst_dstm= self._sess.run(None, {'in_face:0': img, 'morph_value:0':np.float32([morph_value]) })



        for x in [bgr, mask_src_dstm, mask_dst_dstm]:
            x = nn.to_data_format(x,"NHWC", self.model_data_format).astype(np.float32)
            if x.dtype == np.uint8:
                x = x.astype(np.float32)
            x /= 255.0

        return bgr[0], mask_src_dstm[0][...,0], mask_dst_dstm[0][...,0]

    #override
    def get_MergerConfig(self):

        if self._model_type == 2:
            def predictor_morph(face, func_morph_factor=1.0):
                return self.predictor_func(face, func_morph_factor)

            import merger
            return predictor_morph, (self._input_height, self._input_width , 3), merger.MergerConfigMasked(face_type=self.face_type, default_mode = 'overlay', is_morphable=True)
        else:
            import merger
            return self.predictor_func, (self._input_height, self._input_width , 3), merger.MergerConfigMasked(face_type=self.face_type, default_mode = 'overlay')


Model = DFMModel
