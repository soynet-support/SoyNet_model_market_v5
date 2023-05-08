import os
import numpy as np
import cv2
import time
import random


from SoyNet import *
soynet_home = "../.."

def maskrcnn_img():
    #source_input = ["../../data/panda.jpg", "../../data/panda.jpg","../../data/panda.jpg", "../../data/panda.jpg"]
    source_input = ["../../data/panda.jpg"]

    model_code = 'maskrcnn101'
    batch_size = np.size(source_input)
    make_engine = 0
    nms_count = 100
    count_per_class = 10
    img = cv2.imread(source_input[0])
    input_height = img.shape[0]
    input_width = img.shape[1]
    class_count = coco_names.size
    re_height, re_width, pT, pB, pL, pR = calc_maskrcnn_resized_img_dims(720, 1280)

    cfg_file = soynet_home + '/mgmt/configs/%s.cfg'%(model_code)
    license_file = soynet_home + '/mgmt/licenses/license_trial.key'
    log_dir = soynet_home + '/mgmt/logs'
    plugin_dir = soynet_home + '/lib/plugins/'
    weight_file = soynet_home + '/mgmt/weights/%s.weights'%(model_code)
    engine_file = soynet_home + '/mgmt/engines/%s.bin'%(model_code)
    dict_file = soynet_home + "/layer_dict_V5.1.0.dct"
    cache_file = soynet_home + "/mgmt/engines/%s.cache"%(model_code)

    colors = makeColors(class_count)
    thickness=2
    lineType=8
    shift=0


    extend_param = "BATCH_SIZE=%d MODEL_CODE=%s MAKE_ENGINE=%d CLASS_COUNT=%d COUNT_PER_CLASS=%d INPUT_HEIGHT=%d INPUT_WIDTH=%d NMS_COUNT=%d RESIZED_SIZE=%d,%d TOP=%d BOTTOM=%d LEFT=%d RIGHT=%d  LICENSE_FILE=%s LOG_DIR=%s PLUGIN_DIR=%s WEIGHT_FILE=%s ENGINE_FILE=%s DICT_FILE=%s CACHE_FILE=%s"%(
        batch_size, model_code, make_engine, class_count, count_per_class, input_height, input_width, nms_count, re_height, re_width, pT, pB, pL, pR, license_file, log_dir, plugin_dir, weight_file, engine_file, dict_file, cache_file)
    handle = initSoyNet(cfg_file, extend_param)

    input = np.empty((batch_size,re_height,re_width,3))

    for i in range(0, batch_size):

        #use cv2
        # img = cv2.imread(source_input[i])
        # img = cv2.resize(img, (re_width, re_height))

        #use cv+PIL
        img = cv2.imread(source_input[i])
        img = cv2.resize(img, (re_width, re_height))
        img = np.asarray(img)
        img = np.expand_dims(img, axis=0)
        input[i] = img

    # preproc_out = np.zeros((1, 3, 768, 1344), dtype=np.float32)


    output1 = np.zeros((batch_size,nms_count),dtype=BBox)
    output2 = np.zeros((batch_size,nms_count,28,28 ), dtype=np.float32)



    ITER = 10
    total_usec = 0
    for i in range(0,ITER):
        start_time = time.perf_counter()
        feedData(handle, 0, input.astype(np.uint8))
        inference(handle)

        # getOutput(handle, 0, preproc_out)
        output1 = np.zeros((batch_size, nms_count), dtype=BBox)
        output2 = np.zeros((batch_size, nms_count, 28, 28), dtype=np.float32)


        getOutput(handle, 0, output1)
        getOutput(handle, 1, output2)

        soynet_time = time.perf_counter()

        dur = 1/(soynet_time - start_time)

        output1 = output1[0]
        output2 = output2[0]

        # sort_index = output.argsort()[:1,:][-1][::-1]
        # sort_value = np.sort(output)[:1,:][-1][::-1]

        # for j in range(0, 5):
        #     print("%10d %4d prob=%.5f %s" % (j, sort_index[j], sort_value[j], coco_names[sort_index[j]]))

        print("%03d %.2f FPS"%(i,dur))
        total_usec += (dur)

    print("average %.2f fps"%(total_usec/ITER))

    if (True):
        img = cv2.imread(source_input[0])
        for idx in range(0, nms_count):
            if (idx < nms_count and output1[idx]['conf'] > 0):
                x1 = int(output1[idx]['x1'])
                y1 = int(output1[idx]['y1'])
                x2 = int(output1[idx]['x2'])
                y2 = int(output1[idx]['y2'])
                conf = output1[idx]['conf']
                obj_id = output1[idx]['obj_id']
                org = (x1, y1 - 3)
                text = '%s %.6f' % (coco_names[int(obj_id)], conf)

                cv2.rectangle(img, (x1, y1), (x2, y2), colors[int(obj_id)], thickness)
                cv2.putText(img, text, org, cv2.FONT_ITALIC, 1, colors[int(obj_id)], thickness=2)

                mask_thres = 0.6
                alpha = 0.4

                w = x2 - x1
                h = y2 - y1
                mask = cv2.resize(output2[idx],(w,h), cv2.INTER_LINEAR)
                _, threshold = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
                threshold = (threshold*255).astype(np.uint8)
                contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, colors[int(obj_id)], 6, offset=(x1,y1))

            else:
               break

        #img = cv2.resize(img, (int(input_width/4), int(input_height/4)))
        cv2.imshow('result', img)
        cv2.waitKey(0)



    freeSoyNet(handle)

def maskrcnn_video():

    #source_input = ["../../data/panda.jpg", "../../data/panda.jpg","../../data/panda.jpg", "../../data/panda.jpg"]
    source_input = ["../../data/video.mp4"]

    model_code = 'maskrcnn101'
    batch_size = len(source_input)

    vcaps = [cv2.VideoCapture(vn) for vn in source_input] #비디오 연결
    # vcaps = [cv2.VideoCapture(0) for vn in source_input] #webcam 연결

    # vcaps[0].set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # vcaps[0].set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # vcaps[0].set(cv2.CAP_PROP_FPS, 60)
    # while True:
    #     ret, frame = vcaps[0].read()
    #     cv2.imshow("asdf", frame)
    #
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #

    make_engine = 0
    nms_count = 100
    count_per_class = 10
    input_height = int(vcaps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_width = int(vcaps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    class_count = coco_names.size
    re_height, re_width, pT, pB, pL, pR = calc_maskrcnn_resized_img_dims(input_height, input_width)

    cfg_file = soynet_home + '/mgmt/configs/%s.cfg'%(model_code)
    license_file = soynet_home + '/mgmt/licenses/license_trial.key'
    log_dir = soynet_home + '/mgmt/logs'
    plugin_dir = soynet_home + '/lib/plugins/'
    weight_file = soynet_home + '/mgmt/weights/%s.weights'%(model_code)
    engine_file = soynet_home + '/mgmt/engines/%s.bin'%(model_code)
    dict_file = soynet_home + "/layer_dict_V5.1.0.dct"
    cache_file = soynet_home + "/mgmt/engines/%s.cache"%(model_code)

    colors = makeColors(class_count)
    thickness=1
    lineType=8
    shift=0


    extend_param = "BATCH_SIZE=%d MODEL_CODE=%s MAKE_ENGINE=%d CLASS_COUNT=%d COUNT_PER_CLASS=%d INPUT_HEIGHT=%d INPUT_WIDTH=%d NMS_COUNT=%d RESIZED_SIZE=%d,%d TOP=%d BOTTOM=%d LEFT=%d RIGHT=%d  LICENSE_FILE=%s LOG_DIR=%s PLUGIN_DIR=%s WEIGHT_FILE=%s ENGINE_FILE=%s DICT_FILE=%s CACHE_FILE=%s"%(
        batch_size, model_code, make_engine, class_count, count_per_class, input_height, input_width, nms_count, re_height, re_width, pT, pB, pL, pR, license_file, log_dir, plugin_dir, weight_file, engine_file, dict_file, cache_file)
    handle = initSoyNet(cfg_file, extend_param)

    org_imgs = np.zeros((batch_size, input_height, input_width, 3), dtype=np.uint8)
    input = np.empty((batch_size,re_height,re_width,3))

    output1 = np.zeros((batch_size,nms_count),dtype=BBox)
    output2 = np.zeros((batch_size,nms_count,28,28 ), dtype=np.float32)


    loop = True
    total_usec = 0
    frame_idx = 0
    while(loop):
        start_time = time.perf_counter()
        for idx in range(len(vcaps)):
            try :
                _, org_imgs[idx] = vcaps[idx].read()
                img = cv2.resize(org_imgs[idx], (re_width, re_height))
                input[idx] = img
            except :
                loop = False
                break
        if loop == False : break



        feedData(handle, 0, input.astype(np.uint8))
        inference(handle)

        # getOutput(handle, 0, preproc_out)
        output1 = np.zeros((batch_size, nms_count), dtype=BBox)
        output2 = np.zeros((batch_size, nms_count, 28, 28), dtype=np.float32)


        getOutput(handle, 0, output1)
        getOutput(handle, 1, output2)

        soynet_time = time.perf_counter()

        dur = (soynet_time - start_time)
        output1 = output1[0]
        output2 = output2[0]

        total_usec += (dur)

        if (True):
            for idx in range(0, nms_count):
                fps = 1 / dur
                fps_text = "fps : %.2f" % (fps)
                cv2.putText(org_imgs[0], fps_text, (5, input_height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255))

                if (idx < nms_count and output1[idx]['conf'] > 0):
                    x1 = int(output1[idx]['x1'])
                    y1 = int(output1[idx]['y1'])
                    x2 = int(output1[idx]['x2'])
                    y2 = int(output1[idx]['y2'])
                    conf = output1[idx]['conf']
                    obj_id = output1[idx]['obj_id']
                    org = (x1, y1 - 3)
                    text = '%s %.6f' % (coco_names[int(obj_id)], conf)

                    cv2.rectangle(org_imgs[0], (x1, y1), (x2, y2), colors[int(obj_id)], thickness)
                    cv2.putText(org_imgs[0], text, org, cv2.FONT_ITALIC, 0.5, colors[int(obj_id)], thickness=1)



                    mask_thres = 0.6
                    alpha = 0.4

                    w = x2 - x1
                    h = y2 - y1
                    mask = cv2.resize(output2[idx],(w,h), cv2.INTER_LINEAR)
                    _, threshold = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
                    threshold = (threshold*255).astype(np.uint8)
                    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(org_imgs[0], contours, -1, colors[int(obj_id)], 2, offset=(x1,y1))
                else:
                   break

            #img = cv2.resize(img, (int(input_width/4), int(input_height/4)))
            cv2.imshow('result', org_imgs[0])
            key = cv2.waitKey(1)
            if key in (27, ord('q'), ord('Q')): loop = False
            if key == 32: cv2.waitKey(0)

        frame_idx += 1

    avg_fps = frame_idx / total_usec
    print( "avg_fps=%.2f"%avg_fps)

    freeSoyNet(handle)


if __name__ =="__main__" :

    if(1):
        maskrcnn_img()
    else:
        maskrcnn_video()


