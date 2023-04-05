
# Yolov5 Model Overview
YOLOv5 is a family of object detection architectures and models pretrained on the COCO dataset, and represents Ultralytics open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.  
  
SoyNet YOLOv5 support YOLOv5l6, YOLOv5m6, YOLOv5s6, and YOLOv5n6.

# Environments   
#### NVIDIA Development Environment
CUDA version that GPU driver supports.
 - CUDA (>= 12.0)
    ※ You need to use .dll and .so files that match CUDA and TensorRT versions. If you want another version, Please contact [SoyNet](https://soynet.io/en/).
#### S/W
 - OS : windows 10 64 bit, windows 11



# Parameters
※ All parameters are already set in the sample code.
 - `extend_param`
      - `soynet_home` : path of soynet root path
      - `batch_size` : This is the batch-size of the data you want to input.
      - `engine_serialize` : Whether or not the engine is created. (default : 0)
         - 0: Load existing engine file.
         - 1: Create engine file from weight file. you need to set value to in following cases.
            - Change extended param.
            - Change weight file.
      - `region_count`: Number to be used for NMS(Non-maximum Suppression)
      - `nms_count`: Number of NMS (Non-maximum Suppression)
      - `count_per_class`: maximum number of output boxes per class
      - `iou_thres`:  maximum IoU for selected boxes
      - `class_count`: Number of classes
      - `result_count`: total number of output box (include batch size)
      - `conf_thres`: Threshold of score(confidence)
      - `cfg_file`: The path to cfg_file.
      - `weight_file`: The path to weight_file.
      - `engine_file`: The path to engine_file.
      - `log_file`:  The path to log_file.


# Start SoyNet Demo Examples
* Please download dll file and include them in bin floder on soynet file. Check [SoyNet_model_market_v5](https://github.com/soynet-support/SoyNet_model_market_v5/releases/tag/bin_v5.1.0)

#### c++
1.  Prepare standard weight file on [ultralytics release](https://github.com/ultralytics/yolov5/releases/tag/v7.0) or your own one.
2.  Convert weight file in step 1 to SoyNet weight file
  - Open weights(https://github.com/soynet-support/SoyNet_model_market_v5/tree/main/Samples/yolov5/weights) folder.
  - Run ww.py on cmd as the following:
  - Example for ww.py:
	- Supported model_name: yolov5l6, yolov5m6, yolov5s6, and yolov5n6
	- Output_file_path: "../../../mgmt/weights/" + file_name
		- File name is set in https://github.com/soynet-support/SoyNet_model_market_v5/blob/main/Samples/yolov5/yolov5.cpp
	- Help is available by typing:
	```python ww.py -h```
  ```python
  python ww.py --model [model_name] --load_path [prepared weight file including path] --weight_path [output file path]
  #python ww.py --model yolov5m6 --load_path ./yolov5m.pt --weight_path ../../../mgmt/weights/yolov5m6r62.weights
  ```
3.  Make sure main.cpp in https://github.com/soynet-support/SoyNet_model_market_v5/tree/main/Samples is set to run the model you want to run:
```c++
'Lists of functions under folder https://github.com/soynet-support/SoyNet_model_market_v5/tree/main/Samples
int yolov5();	
int yolov8();

int main() {
	//Choose the one you want to run
	yolov5();
	//yolov8();
}
```
4. Run

If you cannot create an engine, review the configuration settings again.

Contact [SOYNET](https://soynet.io/#/contact-us).

# Reference
 - [Original Code](https://github.com/ultralytics/yolov5)

# Acknowlegement

Yolov5 is under GNU General Public License. 
See License terms and condition: [License](https://github.com/ultralytics/yolov5/blob/master/LICENSE)
