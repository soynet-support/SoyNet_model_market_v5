import torch
import torch.nn as nn
from numpy import random
import numpy as np
import argparse

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


#기준
def main():
    parser = argparse.ArgumentParser(description="웨이트 추출 옵션 설정")
    parser.add_argument("-l", "--load_path", help="변환할 파이토치, 텐서플로 웨이트 경로") # 모델 eval에서 watch로 model.pth받은 파일
    args = parser.parse_args()
    model_name = "maskrcnn101"
    model_code = '%s'%(model_name)
    pytorch_weight_path = args.load_path
    soynet_weight_path = "../../../mgmt/weights/%s.weights"%(model_name)

    model_name = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    model = model_zoo.get(model_name, trained=True)

    if 0 :
        for idx, (key,value) in enumerate(weights.items()) :
            wv = value.cpu().data.numpy().flatten()
            print(idx, key, value.shape, wv[:3])
        exit()

    with open(soynet_weight_path, 'wb') as f:
        weights = model.state_dict()

        weight_list = [(key, value) for (key, value) in weights.items()]
        dumy = np.array([0] * 10, dtype=np.int32)
        dumy[0]=5
        dumy[1]=1
        dumy[2]=1
        dumy.tofile(f)

        gen_key = [key for idx, (key, val) in enumerate(model.state_dict().items())]
        gen_val = [val for idx, (key, val) in enumerate(model.state_dict().items())]
        start_end = [
            (16,536),(12,16),(8,12),(4,8),(0,4),
            (536,542),(536,542),(536,542),(536,542),(536,542),
            (542,562)
    ]
        for start, end in start_end:
            for idx in range(start, end):
                key = gen_key[idx]
                val = gen_val[idx]
                w = val.cpu().data.numpy()
                w.tofile(f)
                print(idx,key, w.flatten()[:4], w.shape)


        print('웨이트 생성 완료')
        f.close()

if __name__ =='__main__':
    main()