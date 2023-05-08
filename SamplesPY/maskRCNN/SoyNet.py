# https://stackoverflow.com/questions/22582846/ctypes-structure-arrays-in-python
# http://arkainoh.blogspot.com/2018/08/python.ctypes.html
import ctypes
from ctypes import *
import numpy as np
import numpy.ctypeslib as nc
import platform
import os
import random

#if platform.system()=="Windows" : lib=cdll.LoadLibrary("E:\\2.DEV-5.1.0\lib\SoyNetV5.dll")

class BBox(Structure):
	_fields_ = [("x1", c_float),
				("y1", c_float),
				("x2", c_float),
				("y2", c_float),
				("conf", c_float),
				("obj_id", c_float)]

if platform.system()=="Windows" :

	dll_list = os.listdir("../../bin") #set path of dll folder for SoyNetV5.dll needs

	for i in range(1,len(dll_list)):
		locals()['lib'+str(i)] = ctypes.CDLL("../../bin/"+dll_list[i])


	lib=ctypes.CDLL("..\..\lib\SoyNetV5.dll") #set SoNetV5.dll path

else :
    CDLL("libnvinfer.so", RTLD_GLOBAL)
    CDLL("libcudart.so", RTLD_GLOBAL)
    lib = CDLL("libSoyNetV5.so", RTLD_GLOBAL)

coco_names = np.array(["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
                           "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
                           "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                           "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
                           "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
                           "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop",
                           "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
                           "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
])
def makeColors(class_count):
    colors_array = []
    for i in range(0, class_count):
        colors_array.append([random.randint(50, 256), random.randint(50, 256), random.randint(50, 256)])

    return colors_array

def calc_maskrcnn_resized_img_dims(H,W):
    size = 800.0
    max_size = 1333
    scale = size / min(H,W)

    if(H < W):
        newh = size
        neww = scale * W
    else:
        newh = scale * H
        neww = size

    if(max(newh,neww) > max_size):
        scale = max_size * 1.0 / max(newh, neww)
        newh *= scale
        neww *= scale

    re_height = int(newh + 0.5)
    re_width = int(neww + 0.5)

    stride = 32
    pad_height = int((re_height+ (stride-1)) / stride ) * stride
    pad_width = int((re_width + (stride - 1)) / stride) * stride

    pT = 0
    pB = pad_height - re_height
    pL = 0
    pR = pad_width - re_width

    return re_height, re_width, pT, pB, pL, pR

lib.initSoyNet.argtypes = [c_char_p, c_char_p]
lib.initSoyNet.restype = c_void_p
def initSoyNet(cfg, extent_params="") :
    if extent_params is None : extent_params=""
    return lib.initSoyNet(cfg.encode("utf8"), extent_params.encode("utf8"))

NP_BBox = nc.ndpointer(dtype=BBox, flags="C_CONTIGUOUS")
NP_F32 = nc.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
NP_U8 = nc.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")

lib.feedData.argtypes=[c_void_p, c_ulong, NP_U8]
lib.feedData.restype=None
def feedData(handle, index, data) :
    
    lib.feedData(handle, index, data)

lib.inference.argtypes=[c_void_p]
lib.inference.restype=None
inference = lib.inference

lib.getOutput.argtypes=[c_void_p, c_ulong, NP_F32]
lib.getOutput.restype=None
def getOutput(handle, index, output) :
	if(index==0):
		lib.getOutput.argtypes=[c_void_p, c_ulong, NP_BBox]
	else:
		lib.getOutput.argtypes=[c_void_p, c_ulong, NP_F32]
	lib.getOutput(handle, index, output)

lib.freeSoyNet.argtypes=[c_void_p]
lib.freeSoyNet.restype=None
freeSoyNet = lib.freeSoyNet
