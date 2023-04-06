
# YOLOv8 Model Overview
YOLOv8 is a family of object detection architectures and models pretrained on the COCO dataset, and represents Ultralytics open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.

SoyNet YOLOv8 support YOLOv8l, YOLOv8m, YOLOv8n, and YOLOv8s.



# Environments   
#### NVIDIA Development Environment
CUDA version that GPU driver supports.
 - CUDA (>= 12.0) (Cuda version that gpu driver supported.)
 <br/>※ If you want another version, Please contact [SoyNet](https://soynet.io/en/).
#### S/W
 - OS : windows 10 64 bit, windows 11



# Parameters
  ※ All parameters are already set in the sample code.
 - `soynet_home` : path of soynet root path
 - `extend_param`
      -  `make_engine` : Whether or not the engine is created. (default : 0)
         - 0: Load existing engine file.
         - 1: Create engine file from weight file. you need to set value to in following cases.
            - Change extended param.
            - Change weight file.
      - `batch_size` : This is the batch-size of the data you want to input.
      - `stride` : the ratio by which it downsamples the input
      - `new_height, new_width`
      - `model_name` : YOLOv8 model (SoyNet YOLOv8 support YOLOv8l, YOLOv8m, YOLOv8n, and YOLOv8s)
      - `cfg_file`: The path to cfg_file. it's changed by what you set in model_name.
      - `engine_file`: The path to engine_file.
      - `weight_file`: The path to weight_file.
      - `class_count`: Number of classes.
      - `conf_thres`: Threshold of score(confidence)
      - `iou_thres`:  maximum IoU for selected boxes
      - `region_count`: Number to be used for NMS(Non-maximum Suppression)
      - `count_per_class`: maximum number of output boxes per class
      - `result_count`: total number of output box (include batch size). prefer not to change.


# Start SoyNet Demo Examples

### Installation
* Please download dll file and include them in bin floder on SoyNet file. Follow step in [SoyNet_model_market_v5](https://github.com/soynet-support/SoyNet_model_market_v5/releases/tag/SoyNet_v5.1.0)

#### python
##### Install
Make python environment. python version is not that matter.
```python
pip install opencv-python
```

#### Convert weight file to Soynet weight file
* Skip this setup, if you have Soynet weight file or [SoyNet_model_markey_v5](https://github.com/soynet-support/SoyNet_model_market_v5) already have YOLOv8m soynet weights.
1.  Prepare standard weight file on [ultralytics](https://github.com/ultralytics/ultralytics#models) or your own one.
2.  Convert weight file in step 1 to SoyNet weight file
  > - Open [weights folder](https://github.com/soynet-support/SoyNet_model_market_v5/tree/main/SamplesPY/YOLOv8/Weights).
  >	- set environment to running ww.py 
  >	```python
  >	pip install -r requirements.txt  # install 
  >	```
  > - Run ww.py on cmd as the following:
  >>	```python
  >>	python ww.py --model [model_name]
  >>	```
  >	- Supported model_name: YOLOv8l, YOLOv8m, YOLOv8n, and YOLOv8s
  >	- Output_file_path: "../../../mgmt/weights/%s.weights" %(file_name)
  >     - File name is set in [yolov8.py](https://github.com/soynet-support/SoyNet_model_market_v5/tree/main/SamplesPY/YOLOv8)
  >	- Help is available by typing:
  >		```python ww.py -h```
  >   - Example for ww.py:
  > 		```
  > 		python ww.py --model yolov8m
  > 		```

#### How to run
```python
python yolov8.py
```
# Reference
 - [Original Code](https://github.com/ultralytics/ultralytics)

# Acknowlegement

YOLOv5 is under GNU General Public License. 
See License terms and condition: [License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
