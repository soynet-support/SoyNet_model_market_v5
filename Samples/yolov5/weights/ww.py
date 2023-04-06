import torch
import torch.nn as nn
from numpy import random
import numpy as np
from argparse import ArgumentParser
import sys


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output

def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        #attempt_download(w)
        ckpt = torch.load(w, map_location=map_location)  # load
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble

def main():
    parser = ArgumentParser()
    parser.add_argument("--model", required=True, help="yolov5l6, yolov5m6, yolov5s6, yolov5n6", type=str, default='yolov5l6')
    # parser.add_argument("--load_path", required=True, help="path of orignal weight", type=str, default='./yolov5l6.pt')
    # parser.add_argument("--weight_path", required=True, help="path of saving Soynet weight file", type=str, default='../../../mgmt/weights/yolo.weights')
    args = parser.parse_args()

    model_name = args.model
    pytorch_weight_path = "%s.pt"%(model_name)
    soynet_weight_path = "../../../mgmt/weights/%s.weights"%(model_name)
    sys.path.insert(0, "./yolov5r61")
    
    with open(soynet_weight_path, 'wb') as f:
        checkpoint = torch.load(pytorch_weight_path)

        model = attempt_load(pytorch_weight_path)  # pytorch trained weight

        weights = model.state_dict()

        if 0 :
            for idx, (key,value) in enumerate(weights.items()) :
                wv = value.cpu().data.numpy().flatten()
                print(idx, key, value.shape, wv[:3])
            exit()

        weight_list = [(key, value) for (key, value) in weights.items()]
        dumy = np.array([0] * 10, dtype=np.int32)
        dumy[0]=5
        dumy[1]=1
        dumy[2]=1
        dumy.tofile(f)

        index_l = [0, 6, 10, 22, 6, 10, 22, 26, 30, 54, 26, 30, 54, 58, 62, 98, 58, 62, 98, 102,
                106, 118, 102, 106, 118, 122, 126, 138, 122, 126, 138, 146, 150, 162, 146, 150,
                162, 166, 170, 182, 166, 170, 182, 186, 190, 202, 186, 190, 202, 206,
                210, 222, 206, 210, 222, 226, 230, 242, 226, 230, 242, 246, 250, 262, 246, 250,
                263, 271]

        index_m = [0, 6, 10, 18, 6, 10, 18, 22, 26, 42, 22, 26, 42, 46, 50, 74, 46, 50, 74, 78,
                82, 90, 78, 82, 90, 94, 98, 106, 94, 98, 106, 114, 118, 126, 114, 118, 126, 130,
                134, 142, 130, 134, 142, 146, 150, 158, 146, 150, 158, 162, 166, 174, 162, 166,
                174, 178, 182, 190, 178, 182, 190, 194, 198, 206, 194, 198, 207, 215]  # anchors는 안씀

        index_s = [0, 6, 10, 14, 6, 10, 14, 18, 22, 30, 18, 22, 30, 34, 38, 50, 34, 38, 50, 54,
                58, 62, 54, 58, 62, 66, 70, 74, 66, 70, 74, 82, 86, 90, 82, 86,
                90, 94, 98, 102, 94, 98, 102, 106, 110, 114, 106, 110, 114, 118,
                122, 126, 118, 122, 126, 130, 134, 138, 130, 134, 138, 142, 146, 150, 142, 146, 151, 159]

        index_n = [0, 6, 10, 14, 6, 10, 14, 18, 22, 30, 18, 22, 30, 34, 38, 50, 34, 38, 50, 54,
                58, 62, 54, 58, 62, 66, 70, 74, 66, 70, 74, 82, 86, 90, 82, 86,
                90, 94, 98, 102, 94, 98, 102, 106, 110, 114, 106, 110, 114, 118,
                122, 126, 118, 122, 126, 130, 134, 138, 130, 134, 138, 142, 146, 150, 142, 146, 151, 159]


        ## release V6.2 기준
        if model_name == "yolov5l6":
            index = index_l
        elif model_name == "yolov5m6":
            index = index_m
        elif model_name == "yolov5s6":
            index = index_s
        elif model_name == "yolov5n6":
            index = index_n

        for i_idx in range(int(len(index) / 2)):
            for idx in range(index[i_idx * 2], index[i_idx * 2 + 1]):  #
                key, w = weight_list[idx]
                w = w.cpu().data.numpy()
                w.tofile(f)
                print(0, idx, key, w.shape, w.flatten()[:3])




if __name__ == '__main__':
    main()

