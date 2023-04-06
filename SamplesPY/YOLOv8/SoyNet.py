# https://stackoverflow.com/questions/22582846/ctypes-structure-arrays-in-python
# http://arkainoh.blogspot.com/2018/08/python.ctypes.html

from ctypes import *
import numpy as np
import numpy.ctypeslib as nc

import platform
import os
if platform.system()=="Linux" :
    CDLL("libnvinfer.so", RTLD_GLOBAL)
    CDLL("libcudart.so", RTLD_GLOBAL)
    lib=CDLL("libSoyNetV5.so", RTLD_GLOBAL)
else : # "Windows"
    HOME = os.path.dirname(os.path.realpath(__file__)) + "\\..\\.."
    os.add_dll_directory(HOME + "\\bin")
    os.add_dll_directory(HOME + "\\lib")
    os.environ["PATH"] = HOME + "\\bin;" + HOME + "\\lib;" + os.environ["PATH"]
    lib=cdll.LoadLibrary("SoyNetV5.dll")

coco_label = [ #"BG",
	"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]
colors = [
	0,252,255,0,118,255,0,70,255,0,67,255,0,10,255,0,6,255,0,0,255,0,204,255,0,185,255,0,143,255,0,134,255,0,214,255,0,45,255,0,131,255,0,220,255,0,207,255,0,112,255,0,77,255,0,194,255,0,175,255,0,38,255,0,64,255,0,3,255,0,48,255,0,54,255,0,163,255,0,41,255,0,153,255,0,172,255,0,102,255,0,239,255,0,223,255,0,51,255,0,99,255,0,182,255,0,61,255,0,89,255,0,198,255,0,191,255,0,92,255,0,179,255,0,140,255,0,86,255,0,188,255,0,150,255,0,105,255,0,159,255,0,83,255,0,226,255,0,115,255,0,166,255,0,108,255,0,242,255,0,35,255,0,96,255,0,13,255,0,16,255,0,156,255,0,32,255,0,26,255,0,201,255,0,73,255,0,230,255,0,128,255,0,245,255,0,137,255,0,147,255,0,29,255,0,19,255,0,121,255,0,236,255,0,249,255,0,169,255,0,233,255,0,124,255,0,217,255,0,22,255,0,57,255,0,210,255,0,80,255,
]
class YOLO_SIZE(Structure):
    _pack_ = 1
    _fields_ = [("unpad_height", c_int), ("unpad_width", c_int),("pad_top", c_int),("pad_bottom", c_int),("pad_left", c_int),("pad_right", c_int),]

class YOLO_RESULT(Structure):
    _pack_ = 1
    _fields_ = [("batch_idx", c_int), ("x1", c_int),("y1", c_int),("x2", c_int),("y2", c_int),("id", c_int),("conf", c_float)]

lib.initSoyNet.argtypes = [c_char_p, c_char_p]
lib.initSoyNet.restype = c_void_p
def initSoyNet(cfg, extent_params) :
    if extent_params is None : extent_params=""
    return lib.initSoyNet(cfg.encode("utf8"), extent_params.encode("utf8"))

U8 = nc.ndpointer(dtype=np.uint8, ndim=None, flags='aligned, c_contiguous')
YR = nc.ndpointer(dtype=YOLO_RESULT, ndim=None, flags='aligned, c_contiguous')

def feedData(handle, idx, data) :
    lib.feedData.argtypes = [c_void_p, c_int, U8]
    lib.feedData.restype = None
    lib.feedData(handle,idx, data)

lib.inference.argtypes=[c_void_p]
lib.inference.restype=None
inference = lib.inference

def getOutput(handle, idx, output) :
    lib.getOutput.argtypes = [c_void_p, c_int, YR]
    lib.getOutput.restype = None
    lib.getOutput(handle, idx, output)

lib.calc_resize_yolo.argtypes=[c_void_p, c_int, c_int, c_int, c_int, c_int]
lib.calc_resize_yolo.restype=c_int
def calc_resize_yolo(nH, nW, H, W, stride ) :
    ys = YOLO_SIZE()
    ret = lib.calc_resize_yolo(byref(ys), nH, nW, H, W, stride)
    return ret, ys

lib.freeSoyNet.argtypes=[c_void_p]
lib.freeSoyNet.restype=None
freeSoyNet = lib.freeSoyNet