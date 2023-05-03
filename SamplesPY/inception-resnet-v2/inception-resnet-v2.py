import os
import numpy as np
import cv2
import time
from SoyNet import *
import sys


soynet_home = "../.."
if __name__ =="__main__" :


    #source_input = ["../../data/panda.jpg", "../../data/panda.jpg","../../data/panda.jpg", "../../data/panda.jpg"]
    source_input = [soynet_home + "/data/panda.jpg"]

    img = cv2.imread(source_input[0])
    img = cv2.resize(img, (299, 299))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    np.set_printoptions(threshold=sys.maxsize)

    model_code = 'inception-resnet-v2'
    batch_size = np.size(source_input)
    make_engine = 0
    input_height = 299
    input_width = 299
    class_count = 1000

    cfg_file = soynet_home + '/mgmt/configs/%s.cfg'%(model_code)
    license_file = soynet_home + '/mgmt/licenses/license_trial.key'
    log_dir = soynet_home + '/mgmt/logs'
    plugin_dir = soynet_home + '/lib/plugins/'
    weight_file = soynet_home + '/mgmt/weights/%s.weights'%(model_code)
    engine_file = soynet_home + '/mgmt/engines/%s.bin'%(model_code)
    dict_file = soynet_home + "/layer_dict_V5.1.0.dct"
    cache_file = soynet_home + "/mgmt/engines/%s.cache"%(model_code)

    extend_param = "BATCH_SIZE=%d MAKE_ENGINE=%d MODEL_CODE=%s INPUT_SIZE=%d,%d CLASS_COUNT=%d LICENSE_FILE=%s LOG_DIR=%s PLUGIN_DIR=%s WEIGHT_FILE=%s ENGINE_FILE=%s DICT_FILE=%s CACHE_FILE=%s"%(
        batch_size, make_engine, model_code, input_height, input_width, class_count, license_file, log_dir, plugin_dir, weight_file, engine_file, dict_file, cache_file)
    handle = initSoyNet(cfg_file, extend_param)

    input = np.empty((batch_size,299,299,3))

    for i in range(0, batch_size):
        img = cv2.imread(source_input[i])
        img = cv2.resize(img, (299, 299))
        img = np.asarray(img)
        img = np.expand_dims(img, axis=0)
        input[i] = img

    output = np.zeros((batch_size,class_count+1),dtype=np.float32)
    ITER = 100
    total_usec = 0
    for i in range(0,ITER):
        start_time = time.perf_counter()
        feedData(handle, 0, input.astype(np.uint8))
        inference(handle)
        getOutput(handle, 0, output)

        soynet_time = time.perf_counter()

        dur = 1/(soynet_time - start_time)

        sort_index = output.argsort()[:1,:][-1][::-1]
        sort_value = np.sort(output)[:1,:][-1][::-1]

        for j in range(0, 5):
            print("%10d %4d prob=%.5f %s" % (j, sort_index[j], sort_value[j], class_name[sort_index[j]]))

        print("%03d %.2f FPS"%(i,dur))
        total_usec += (dur)

    print("average %.2f fps"%(total_usec/ITER))

    freeSoyNet(handle)


