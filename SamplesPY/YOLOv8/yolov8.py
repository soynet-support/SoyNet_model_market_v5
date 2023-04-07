import numpy as np
import cv2
import time
import platform

from SoyNet import *
soynet_home = "../.."

#video_names = [ soynet_home + "/data/NY.mkv", soynet_home + "/data/po.mp4", soynet_home + "/data/NY.mkv", soynet_home + "/data/po.mp4", soynet_home + "/data/NY.mkv", soynet_home + "/data/po.mp4" ] # 동일한 크기
#video_names = [ soynet_home + "/data/NY.mkv", soynet_home + "/data/po.mp4" ]
video_names = [ soynet_home + "/data/video.mp4" ]

make_engine = 0

batch_size = len(video_names)

vcaps = [cv2.VideoCapture(vn) for vn in video_names ]
org_height = int(vcaps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
org_width  = int(vcaps[0].get(cv2.CAP_PROP_FRAME_WIDTH))

stride = 32
new_height, new_width = (640, 640)
ret, ys = calc_resize_yolo(new_height, new_width,org_height,org_width,stride)

model_name = "yolov8s"
cfg_file = soynet_home + "/mgmt/configs/%s.cfg"%model_name
engine_file = soynet_home + "/mgmt/engines/%s.bin" % model_name
weight_file = soynet_home + "/mgmt/weights/%s.weights" % model_name
if platform.system() == "Linux":
    plugin  = "plugins_ubuntu_20.04"
elif platform.system() == "Windows":
    plugin = "plugins_windows"

class_count = len(coco_label)
conf_thres=0.25
iou_thres=0.7
region_count = 2000
count_per_class = 30
result_count = batch_size * count_per_class * class_count
extend_param = "BATCH_SIZE=%d SOYNET_HOME=%s MODEL_NAME=%s MAKE_ENGINE=%d ENGINE_FILE=%s PLUGIN_DIR=%s CLASS_COUNT=%d CONF_THRES=%f IOU_THRES=%f REGION_COUNT=%d COUNT_PER_CLASS=%d RESULT_COUNT=%d WEIGHT_FILE=%s RE_SIZE=%d,%d ORG_SIZE=%d,%d TOP=%d BOTTOM=%d LEFT=%d RIGHT=%d"%(
        batch_size, soynet_home, model_name, make_engine, engine_file, plugin, class_count, conf_thres, iou_thres, region_count,
        count_per_class, result_count, weight_file, ys.unpad_height, ys.unpad_width,
        org_height, org_width, ys.pad_top, ys.pad_bottom, ys.pad_left, ys.pad_right)

if 1: # initSoyNet
    handle = initSoyNet(cfg_file, extend_param)  # handle_0가 사용한 working 메모리 재사용

if 1 : # inference loop
    org_imgs = np.zeros(( batch_size, org_height, org_width, 3), dtype=np.uint8)
    input = np.zeros(( batch_size, ys.unpad_height, ys.unpad_width, 3), dtype=np.uint8)

    loop = True
    frame_idx = 0
    total_dur = 0
    while(loop) : # loop for inference
        start = time.perf_counter()

        for idx in range(len(vcaps)) :
            try :
                _, org_imgs[idx] = vcaps[idx].read()
                img = cv2.resize(org_imgs[idx], (ys.unpad_width, ys.unpad_height))
                input[idx] = img
            except :
                loop = False
                break
        if loop == False : break

        feedData(handle, 0, input)
        inference(handle)

        output = np.zeros((result_count,), dtype=YOLO_RESULT) # 출력 공간을 clear하여 생성해야 함
        getOutput(handle, 0, output)

        end = time.perf_counter()
        dur = (end - start)
        total_dur += dur

        if 1 : # display
            for batch_idx, x1, y1, x2, y2, obj_id, conf in output :
                if conf == 0. : break

                if 1 :
                    print("frame_idx=%d batch_idx=%d %4d %4d %4d %4d %2d %-15s %.4f"%(frame_idx, batch_idx, x1, y1, x2, y2, obj_id, coco_label[obj_id], conf))

                if 1 :
                    color = colors[obj_id*3: obj_id*3+3]
                    cv2.rectangle( org_imgs[batch_idx], (x1,y1), (x2,y2), color, thickness = 2, lineType = 8, shift = 0 )
                    text = coco_label[obj_id] + " %.4f"%(conf)
                    cv2.putText(org_imgs[batch_idx], text, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
            fps = 1 / dur
            fps_text = "fps : %.2f"%(fps)
            for idx in range(batch_size) :
                cv2.putText(org_imgs[idx], fps_text, (5, org_height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                win_name = "%d_%s"%(idx,video_names[idx])
                cv2.imshow( win_name, org_imgs[idx])
            key = cv2.waitKey(1)
            if key in (27 , ord('q') , ord('Q')) : loop=False
            if key == 32: cv2.waitKey(0)

        frame_idx +=1

    avg_fps = frame_idx / total_dur
    print( "avg_fps=%.2f"%avg_fps)

if 1 : # freeSoyNet
    freeSoyNet(handle)
