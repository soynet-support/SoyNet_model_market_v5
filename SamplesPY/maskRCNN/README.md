# Mask-RCNN Model Overview
The Mask R-CNN has a structure that adds a mask branch to the Fast R-CNN that predicts the segmentation mask. In order to perform segmentation tasks more effectively, the paper added a RoIallign layer that preserves the specific location of the object.


# Environments   

#### S/W
 - OS : Ubuntu 20.04 LTS

#### NVIDIA Development Environment
- TensorRT (= 8.6.0)
`pip install tensorrt`
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
 - `extend_param`
      -  `make_engine` : Whether or not the engine is created. (default : 0)
         - 0: Load existing engine file.
         - 1: Create engine file from weight file. you need to set value to in following cases.
            - Change extended param.
            - Change weight file.
      - `batch_size` : This is the batch-size of the data you want to input.
      - `model_code` : Inception-resnet-v2
      - `class_count`: Number of classes
      - `count_per_class`: maximum number of output boxes per class
      - `input_height`, `input_width`: Input size of maskRCNN
      - 'nms_count`: Maximnum number of nms
      - `re_height`, `re_width`: Resize of input size. It is automatically calculated.
      - `pT`,`pB`,`pL`,`pR`: Padding size. It is automatically calculated.
      - `license_file`: The path to license_file.
      - `log_dir`: The path to log_dir.
      - `plugin_dir`: The path to plugin_dir.
      - `weight_file`: The path to weight_file.
      - `engine_file`: The path to engine_file.
      - `dict_file`: The path to dict_file.
      - `cache_file`: the path to cache_file.

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

1.  Prepare standard weight file from torch.
2.  Convert weight file in step 1 to SoyNet weight file
  > - Open [weights folder](https://github.com/soynet-support/SoyNet_model_market_v5/tree/main/SamplesPY/maskRCNN/Weights).
  >	- set environment to running ww.py 
  >	```python 
  >	pip install -r requirements.txt  # install 
  >	```
  > - Run ww.py on cmd as the following:
  >>	```python
  >>	python ww.py --load_path [path of original weight]
  >>	```
  >	- Output_file_path: "../../../mgmt/weights/%s.weights" %(file_name)
  >     - File name is set in [maskRCNN.py](https://github.com/soynet-support/SoyNet_model_market_v5/tree/main/SamplesPY/maskRCNN)
  >	- Help is available by typing:
  >		```python ww.py -h```
  >   - Example for ww.py:
  > 		```
  > 		python ww.py --load_path ./chekpoint.pth
  > 		```

#### How to run
```python
export LD_LIBRARY_PATH=../../lib:$LD_LIBRARY_PATH
python inception-resnet-v2.py
```
# Reference
 - [Original Code](https://github.com/facebookresearch/detectron2)

# Acknowlegement

maskRCNN is under Apache License. 
See License terms and condition: [License](https://github.com/facebookresearch/detectron2/blob/main/LICENSE)
