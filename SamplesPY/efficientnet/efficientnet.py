
import os
import numpy as np
import cv2
import time
from SoyNet import *

model_sizes = [224,240,260,300,380,456,528,600]
model_resizes = [254,272,295,340,431,456,599,681]

soynet_home = "../.."

if __name__ =="__main__" :

    #source_input = ["../../data/panda.jpg", "../../data/panda.jpg","../../data/panda.jpg", "../../data/panda.jpg"]
    source_input = [soynet_home + "/data/panda.jpg"]

    model_code = Efficientnet.B7.value
    make_engine = 1
    class_count = 1000
    batch_size = np.size(source_input)
    input_height = model_sizes[model_code]
    input_width = model_sizes[model_code]
    re_width = model_resizes[model_code]

    model_name = 'efficientnet-b%d'%(model_code)
    cfg_file = soynet_home + '/mgmt/configs/%s.cfg'%(model_name)
    license_file = soynet_home + '/mgmt/licenses/license_trial.key'
    log_dir = soynet_home + '/mgmt/logs'
    plugin_dir = soynet_home + '/lib/plugins/'
    weight_file = soynet_home + '/mgmt/weights/%s.weights'%(model_name)
    engine_file = soynet_home + '/mgmt/engines/%s.bin'%(model_name)
    dict_file = soynet_home + "/layer_dict_V5.1.0.dct"
    cache_file = soynet_home + "/mgmt/engines/%s.cache"%(model_name)

    extend_param = "BATCH_SIZE=%d MAKE_ENGINE=%d MODEL_CODE=%s INPUT_SIZE=%d,%d CLASS_COUNT=%d LICENSE_FILE=%s LOG_DIR=%s PLUGIN_DIR=%s WEIGHT_FILE=%s ENGINE_FILE=%s DICT_FILE=%s CACHE_FILE=%s"%(
        batch_size, make_engine, model_name, input_height, input_width, class_count, license_file, log_dir, plugin_dir, weight_file, engine_file, dict_file, cache_file)
    handle = initSoyNet(cfg_file, extend_param)

    input = np.empty((batch_size,input_height,input_width,3),dtype=np.uint8)

    for i in range(0, batch_size):
        img = cv2.imread(source_input[i])
        img = cv2.resize(img, (re_width, model_sizes[model_code]))
        x,y = int((model_resizes[model_code] - model_sizes[model_code]) / 2),0
        w,h = model_sizes[model_code], model_sizes[model_code]
        img = img[y:y + h, x:x + w]
        img = np.asarray(img)
        img = np.expand_dims(img, axis=0)
        input[i] = img

    output = np.zeros((batch_size,class_count),dtype=np.float32)
    ITER = 200
    total_usec = 0
    for i in range(0,ITER):
        start_time = time.perf_counter()
        feedData(handle, 0, input)
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


