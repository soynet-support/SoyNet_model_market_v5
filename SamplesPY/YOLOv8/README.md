
# YOLOv8 Model Overview
YOLOv8 is a family of object detection architectures and models pretrained on the COCO dataset, and represents Ultralytics open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.

SoyNet YOLOv8 support YOLOv8l, YOLOv8m, YOLOv8n, and YOLOv8s.



# Environments   

#### S/W
 - OS : Ubuntu 20.04 LTS

#### NVIDIA Development Environment
- TensorRT (= 8.6.0)
 <details close>
<summary>MORE INFO</summary>
Actually, SoyNet needs only

- libnvinfer.so
- libnvinfer_builder_resource.so.8.6.0

If you need an efficient way to run SoyNet, just include those files (from TensorRT SDK) in the lib folder.

But due to Nvidia license policy, we cannot provide a portion of the TensorRT SDK.

So technically, SoyNet doesn't need to install the TensorRT SDK.
##### More about [SoyNet](https://soynet.io/).
</details>

CUDA version that GPU driver supports.
 - CUDA Driver Version (>= 12.0) (Cuda version that gpu driver supported.)
<details open>
<summary>How To Check Nvidia GPU Driver</summary>

Use command line and consult nvidia-smi utility to reveal NVIDIA driver version:
```cmd
nvidia-smi
```
+-----------------------------------------------------------------------------+
<br/>| NVIDIA-SMI 525.78.01    Driver Version: 525.78.01    CUDA Version: 12.0        |
<br/>|-------------------------------+----------------------+----------------------+


</details>

 
 <br/>※ If you want another version, Please contact [SoyNet](https://soynet.io/en/).

# Parameters
  ※ All parameters are already set in the sample code.
 - `soynet_home` : path of SoyNet root path
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
### python
##### Install
Make python environment. Python version does not matter.

#### Download pre-trained weights files (already converted for SoyNet)
```python
cd ./Weights
python download_soynet_weight.py
```
* default of download path is `../../../mgmt/weights` but if you want to set download path, you can command `python download_soynet_weight.py --path [path]`
#### Convert weight file to SoyNet weight file
* You can skip this step, if you already have SoyNet weight file 

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

YOLOv8 is under GNU General Public License. 
See License terms and condition: [License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
