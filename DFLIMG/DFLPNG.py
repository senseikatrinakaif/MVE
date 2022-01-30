from json import load
import struct
import traceback

import cv2
import numpy as np

from core import imagelib
from core.cv2ex import *

from core.imagelib import SegIEPolys
from io import BytesIO
from core.interact import interact as io
from core.structex import *
from facelib import FaceType

import base64


from zlib import crc32, compress, decompress
from ast import literal_eval

itxt_key = b"dflabmve"

import json

# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)

class NDArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            output = BytesIO()
            np.savez_compressed(output, obj=obj)
            return {'b64npz' : base64.b64encode(output.getvalue()).decode('utf-8')}
        return json.JSONEncoder.default(self, obj)




class DFLPNG(object):
    def __init__(self, filename):
        self.filename = filename
        self.data = b""
        self.length = 0
        self.chunks = []
        self.dfl_dict = None
        self.shape = None
        self.img = None
        self.shape = None
        self.img = None



    @staticmethod
    def load_raw(filename, loader_func=None):
        try:
            if loader_func is not None:
                data = loader_func(filename)
            else:
                with open(filename, "rb") as f:
                    data = f.read()
        except:
            raise FileNotFoundError(filename)

        try:
            inst = DFLPNG(filename)
            inst.data = data
            inst.length = len(data)
            

            return inst
        except Exception as e:
            raise Exception (f"Corrupted PNG file {filename} {e}")

    @staticmethod
    def load(filename, loader_func=None):
        try:
            inst = DFLPNG.load_raw (filename, loader_func=loader_func)
            inst.dfl_dict = {}
            
            loaded_dict = DFLPNG.png_read_meta(inst.data)

            if loaded_dict != None:
                inst.dfl_dict = loaded_dict.copy()
                if loaded_dict.get("xseg_mask", None) is not None:
                    inst.dfl_dict.pop("xseg_mask")
                    mask = loaded_dict["xseg_mask"]
                    inst.set_xseg_mask(mask)



            return inst
        except Exception as e:
            io.log_err (f'Exception occured while DFLPNG.load : {traceback.format_exc()}')
            return None

    def has_data(self):
        return len(self.dfl_dict.keys()) != 0

    def save(self):
        try:
            with open(self.filename, "wb") as f:
                f.write ( self.dump() )
        except:
            raise Exception( f'cannot save {self.filename}' )

    def dump(self):
        data = b""

        dict_data = self.dfl_dict.copy()

        for key in list(dict_data.keys()):
            #print(key)
            if dict_data[key] is None:
                dict_data.pop(key)
            elif key is "xseg_mask":

                #set uncompressed array 
                mask = self.get_xseg_mask()
                dict_data["xseg_mask"] = mask
                
                
            elif isinstance(dict_data[key], np.ndarray):
                dict_data[key] = dict_data[key].tolist()
                       

        dict_data = json.dumps(dict_data, cls=NDArrayEncoder)
                

        data = self.png_write_meta(self.data, dict_data)
        

        return data

    def get_img(self):
        if self.img is None:
            self.img = cv2_imread(self.filename)
        return self.img

    def get_shape(self):
        if self.shape is None:
            img = self.get_img()
            if img is not None:
                self.shape = img.shape
        return self.shape

    def get_height(self): #TODO
        return self.shape[1]

    def get_dict(self):
        return self.dfl_dict

    def set_dict (self, dict_data=None):
        self.dfl_dict = dict_data

    def get_face_type(self):            return self.dfl_dict.get('face_type', FaceType.toString (FaceType.FULL) )
    def set_face_type(self, face_type): self.dfl_dict['face_type'] = face_type

    def get_landmarks(self):            return np.array ( self.dfl_dict['landmarks'] )
    def set_landmarks(self, landmarks): self.dfl_dict['landmarks'] = landmarks

    def get_eyebrows_expand_mod(self):                      return self.dfl_dict.get ('eyebrows_expand_mod', 1.0)
    def set_eyebrows_expand_mod(self, eyebrows_expand_mod): self.dfl_dict['eyebrows_expand_mod'] = eyebrows_expand_mod

    def get_source_filename(self):                  return self.dfl_dict.get ('source_filename', None)
    def set_source_filename(self, source_filename): self.dfl_dict['source_filename'] = source_filename

    def get_source_rect(self):              return self.dfl_dict.get ('source_rect', None)
    def set_source_rect(self, source_rect): self.dfl_dict['source_rect'] = source_rect

    def get_source_landmarks(self):                     return np.array ( self.dfl_dict.get('source_landmarks', None) )
    def set_source_landmarks(self, source_landmarks):   self.dfl_dict['source_landmarks'] = source_landmarks

    def get_image_to_face_mat(self):
        mat = self.dfl_dict.get ('image_to_face_mat', None)
        if mat is not None:
            return np.array (mat)
        return None
    def set_image_to_face_mat(self, image_to_face_mat):   self.dfl_dict['image_to_face_mat'] = image_to_face_mat

    def has_seg_ie_polys(self):
        return self.dfl_dict.get('seg_ie_polys',None) is not None

    def get_seg_ie_polys(self):
        d = self.dfl_dict.get('seg_ie_polys',None)
        if d is not None:
            d = SegIEPolys.load(d)
        else:
            d = SegIEPolys()

        return d

    def set_seg_ie_polys(self, seg_ie_polys):
        if seg_ie_polys is not None:
            if not isinstance(seg_ie_polys, SegIEPolys):
                raise ValueError('seg_ie_polys should be instance of SegIEPolys')

            if seg_ie_polys.has_polys():
                seg_ie_polys = seg_ie_polys.dump()
            else:
                seg_ie_polys = None

        self.dfl_dict['seg_ie_polys'] = seg_ie_polys

    def has_xseg_mask(self):
        return self.dfl_dict.get('xseg_mask',None) is not None

    def get_xseg_mask_compressed(self):
        mask_buf = self.dfl_dict.get('xseg_mask',None)
        if mask_buf is None:
            return None

        return mask_buf
        
    def get_xseg_mask(self):
        mask_buf = self.dfl_dict.get('xseg_mask',None)
        if mask_buf is None:
            return None

        img = cv2.imdecode(mask_buf, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 2:
            img = img[...,None]

        return img.astype(np.float32) / 255.0


    def set_xseg_mask(self, mask_a):
        if mask_a is None:
            self.dfl_dict['xseg_mask'] = None
            return

        mask_a = imagelib.normalize_channels(mask_a, 1)
        img_data = np.clip( mask_a*255, 0, 255 ).astype(np.uint8)

        data_max_len = 50000

        ret, buf = cv2.imencode('.png', img_data)

        if not ret or len(buf) > data_max_len:
            for jpeg_quality in range(100,-1,-1):
                ret, buf = cv2.imencode( '.jpg', img_data, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality] )
                if ret and len(buf) <= data_max_len:
                    break

        if not ret:
            raise Exception("set_xseg_mask: unable to generate image data for set_xseg_mask")

        self.dfl_dict['xseg_mask'] = buf

    def pack_to_itxt(self,metadata):
        """ Pack the given metadata dictionary to a PNG iTXt header field.
        Parameters
        ----------
        metadata: dict or bytes
            The dictionary to write to the header. Can be pre-encoded as utf-8.
        Returns
        -------
        bytes
            A byte encoded PNG iTXt field, including chunk header and CRC
        """
        if not isinstance(metadata, bytes):
            metadata = str(metadata).encode("utf-8", "strict")
        key = "dflabmve".encode("latin-1", "strict")

        chunk = key + b"\0\0\0\0\0" + metadata
        crc = struct.pack(">I", crc32(chunk, crc32(b"iTXt")) & 0xFFFFFFFF)
        length = struct.pack(">I", len(chunk))
        retval = length + b"iTXt" + chunk + crc
        return retval

    def png_write_meta(self, png, data):
        """ Write Faceswap information to a png's iTXt field.
        Parameters
        ----------
        png: bytes
            The bytes encoded png file to write header data to
        data: dict or bytes
            The dictionary to write to the header. Can be pre-encoded as utf-8.
        Notes
        -----
        This is a fairly stripped down and non-robust header writer to fit a very specific task. OpenCV
        will not write any iTXt headers to the PNG file, so we make the assumption that the only iTXt
        header that exists is the one that we created for storing alignments.
        References
        ----------
        PNG Specification: https://www.w3.org/TR/2003/REC-PNG-20031110/
        """
        
        pointer = png.find(b"iTXt") - 4
        if pointer > 0:

            retval = self.update_existing_metadata(png, data)

        else:
            split = png.find(b"IDAT") - 4
            retval = png[:split] + self.pack_to_itxt(data) + png[split:]
        return retval

    @staticmethod
    def png_read_meta(png):
        """ Read the Faceswap information stored in a png's iTXt field.

        Parameters
        ----------
        png: bytes
            The bytes encoded png file to read header data from

        Returns
        -------
        dict
            The Faceswap information stored in the PNG header

        Notes
        -----
        This is a very stripped down, non-robust and non-secure header reader to fit a very specific
        task. OpenCV will not write any iTXt headers to the PNG file, so we make the assumption that
        the only iTXt header that exists is the one that Faceswap created for storing alignments.
        """
        retval = None
        pointer = 0
        length = 0

        def ndarray_decoder(dct):
            if isinstance(dct, dict) and 'b64npz' in dct:
                output = BytesIO(base64.b64decode(dct['b64npz']))
                output.seek(0)
                return np.load(output)['obj']
            return dct


        while True:
            pointer = png.find(b"iTXt", pointer) - 4

            if pointer < 0:
                #logger.trace("No metadata in png")
                break
            length = struct.unpack(">I", png[pointer:pointer + 4])[0]
            pointer += 8
            keyword, value = png[pointer:pointer + length].split(b"\0", 1)
            if keyword == itxt_key:
                retval = json.loads(value[4:].decode("utf-8"), object_hook=ndarray_decoder)
                break
            #logger.trace("Skipping iTXt chunk: '%s'", keyword.decode("latin-1", "ignore"))
            pointer += length + 4

        return retval


    
    def update_existing_metadata(self, pngbytes, metadata):
        """ Update the png header metadata for an existing .png extracted face file on the filesystem.

        Parameters
        ----------
        png: bytes
                The bytes encoded png file to read header data from
        metadata: dict or bytes
            The dictionary to write to the header. Can be pre-encoded as utf-8.
        Returns
        -------
        bytes
            bytes encoded PNG with added metadata
        """
        with BytesIO(pngbytes) as png,  BytesIO() as tmp:
            chunk = png.read(8)
            if chunk != b"\x89PNG\r\n\x1a\n":
                raise ValueError(f"Invalid header found in png")
            tmp.write(chunk)

            while True:
                chunk = png.read(8)
                length, field = struct.unpack(">I4s", chunk)
                #logger.trace("Read chunk: (chunk: %s, length: %s, field: %s)", chunk, length, field)

                if field == b"IDAT":  # Write out all remaining data
                    #logger.trace("Writing image data and closing png")
                    tmp.write(chunk + png.read())
                    break

                if field != b"iTXt":  # Write non iTXt chunk straight out
                    #logger.trace("Copying existing chunk")
                    tmp.write(chunk + png.read(length + 4))  # Header + CRC
                    continue

                keyword, value = png.read(length).split(b"\0", 1)
                if keyword != itxt_key:
                    # Write existing non fs-iTXt data + CRC
                    #logger.trace("Copying non-faceswap iTXt chunk: %s", keyword)
                    tmp.write(keyword + b"\0" + value + png.read(4))
                    continue

                #logger.trace("Updating faceswap iTXt chunk")
                tmp.write(self.pack_to_itxt(metadata))
                png.seek(4, 1)  # Skip old CRC

            tmp.seek(0)
            return tmp.read()