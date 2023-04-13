![SoyNet](https://user-images.githubusercontent.com/74886743/161455587-31dc85f4-d60c-4dd5-9612-113a9ac82c41.png)
# SoyNet_model_market_v5

[SoyNet](https://soynet.io/) is an inference optimizing solution for AI models.

## SoyNet Overview

#### Core technology of SoyNet

- Accelerate model inference by maximizing the utilization of numerous cores on the GPU without compromising accuracy (2x to 5x compared to Tensorflow or Pytorch).
- Minimize GPU memory usage down to 1/5~1/15 of Tensorflow or Pytorch.

   ※ Performance varies depending on model and/or configuration environment.
   
#### Benefit of SoyNet

- can support customer to provide AI applications and AI services in time. (Time to Market)
- can help application developers to easily execute AI projects without additional technical AI knowledge and experience.
- can help customer to reduce H/W (GPU, GPU server) or Cloud Instance cost for the same AI execution. (inference)
- can support customer to respond to real-time environments that require very low latency in AI inference.
   
#### Features of SoyNet

- Dedicated engine for inference of deep learning models.
- Supports NVIDIA GPUs
- library files to be easiliy integrated with customer applications dll file (Windows)
- We can provide c++ and python executable files.

## Folder Structure


```
   ├─SamplesPy          :  sample code of AI model on python (such as yolov5...)
   |  ├─model
   |  |  ├─weights      
   |  |  |  └─ww.py     : you can make soynet weight file from your own weight file
   |  |  └─model.py    : execution file
   ├─data               : sample data(such as jpg, mp4..) for sample code
   ├─lib                : lib for release
   ├─mgmt               : SoyNet executuion env
   │  ├─configs         : Model definitions (*.cfg)
   │  ├─engines         : SoyNet engine files
   │  ├─licenses        : license file
   │  ├─logs            : SoyNet log files
   │  └─weights         : Weight files for SoyNet models (*.weights)
   └─layer_dict_V5.1.0  : dictionary file for soynet layer
```
 - `engines` : it's made at the first time execution or when you modify the configs file.
 - `weights` : You can make .weights file from your own trained weight file(ex. *.pt) on ww.py in [weights folder](#folder-structure).
 - `license file` : Please contact [SoyNet](https://soynet.io/) if the time has passed.
 
 ## SoyNet Function
 - `initSoyNet(.cfg, extend_param)` : Created a SoyNet handle.
 - `feedData(handle, data)` : Put the data into the SoyNet handle.
 - `inference(handle)` : Start inference.
 - `getOutput(handle, output)` : Put the inference data into the output.
 - `freeSoyNet(handle)` : If you're done using a handle, destroy it.
 
    ※ `extend_param`
      - `extend_param` contains parameters necessary to define the model, such as input size, engine_serialize, batch_size ...
      - The parameters required may vary depending on the model.

   ※ `make_engine` in `extend_param`
      - This parameter determines whether to build a SoyNet engine or not.
      - default is 0.
      - If you run SoyNet for the first time or modify config or extended parameters, select one of the following two methods.
         1) Delete the existing generated bin (engine) file and run it again.
         2) Run by setting this value to 1 and then change back to 0.
         
         
## Prerequisites

#### S/W
 - OS : Ubuntu 20.04 LTS
 - Others : `pip insatll -r requirements.txt`
   - OpenCV (for reading video files and outputting the screen)
   - Wget (for `download_soynet_weight.py` in [weights folder](#folder-structure))
   - Numpy
   
#### NVIDA Development Environment
 
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

#### CUDA version that GPU driver supports.
 - CUDA Driver Version (>= 12.0) (Cuda version that gpu driver supported.)

 
 <details close>
<summary>How To Check Nvidia GPU Driver</summary>

Use command line and consult nvidia-smi utility to reveal NVIDIA driver version:
```cmd
nvidia-smi
```
+-----------------------------------------------------------------------------+
<br/>| NVIDIA-SMI 525.78.01    Driver Version: 525.78.01    CUDA Version: 12.0        |
<br/>|-------------------------------+----------------------+----------------------+


</details>

※ If you want another version, Please contact [SoyNet](https://soynet.io/).
    
#### H/W
 - Tested GPU architecture : 
   - Pascal/Volta/Turing/Ampere/Ada Lovelac (ex: for PC Nvidia GTX 10xx, RTX 20xx/30xx/40xx, etc)
   - Nvidia 384-Core Volta GPU (in Nvidia NX Xavier)
 
 
※ Please contact [SoyNet](https://soynet.io/) for specific GPU support.
    




