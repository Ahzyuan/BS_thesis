# Research and application of vehicle automatic yield pedestrian algorithm based on multi-target detection and tracking

![Static Badge](https://img.shields.io/badge/License-Apache_2.0-red?link=https://www.apache.org/licenses/LICENSE-2.0) ![Static Badge](https://img.shields.io/badge/SCAU-DOIC_2024-green?labelColor=007A49&link=https%3A%2F%2Fmetc.scau.edu.cn%2Fscaudoic2024%2F)

[![Static Badge](https://img.shields.io/badge/Ultralytics-YOLOv8.2-purple)](https://github.com/ultralytics/ultralytics) ![Static Badge](https://img.shields.io/badge/TensorRT-JetsonTX2-green?logo=nvidia&color=76B900&link=https%3A%2F%2Fdeveloper.nvidia.cn%2Ftensorrt&link=https%3A%2F%2Fdeveloper.nvidia.com%2Fembedded%2Fjetson-tx2) ![Static Badge](https://img.shields.io/badge/Intel-RealSense--D435-lightcyan?logo=intel&color=F4F4F4&link=https%3A%2F%2Fwww.intelrealsense.com%2Fdepth-camera-d435%2F)

---

Chinese: [README_zhCN](README_zhCN.md)

## 😀 Introduction

1. Based on [YOLOv8](https://github.com/ultralytics/ultralytics) , this project relies on binocular depth camera `RealSense-D435` and `Jetson-TX2(ARM64-Ubuntu 18.04)` platform to improve the algorithm proposed by the master branch, improve its performance in terms of **stability, accuracy and real-time performance, and realize its deployment at the edge.**

3. This project uses the implementation process and coding style of `YOLOv8` for reference. By modifying the algorithm proposed by `YOLOv8` and the master branch, this project can be compatible with the original `YOLOv8`, while enhancing the encapsulation, the ease using of the module in implementation, and the stability and accuracy of the results in function.

5. Specifically, the **HighLights** of this project are: 
	- **100% compatiable with the origin project**: Still able to automatically yield to pedestrians from the perspective of the driving recorder
	- **More accurate visual distance measurement**: Compatible with the original video and picture input, the binocular camera can be turned on to achieve more accurate visual ranging. At the same time, a variety of advanced setting interfaces of binocular camera, such as automatic exposure, automatic white balance and hole filling technique in depth image, are realized to ensure the robustness of ranging value to environmental changes.
	- **More accurate tracking algorithms**: While being compatible with the original rule-based target tracking algorithm, various professional tracking algorithms can be enabled according to hardware resources, such as [ByteTrack](https://github.com/ifzhang/ByteTrack) and [BoTSORT](https://github.com/viplix3/BoTSORT-cpp).
	- **Higher balance of accuracy and real-time performance**: Through the model transformation path of `pt -> onnx -> tensorRT`, the layer fusion mechanism of `tensorRT` is fully utilized to complete the purification of the model. Using the model converted by` yolov8s.pt` and running the original algorithm + binocular ranging + `ByteTrack` tracking algorithm on the `Jetson-TX2`, the frame rate can still be stable at `28 - 32 fps`.
	- **More intuitive interaction**: Achieve more diverse and richer terminal information feedback through `rich`. At the same time, frame processing information including depth-color image fusion, decision target classification, light color judgment, etc. is easier to obtain, display and save. In addition, a new beep alarm mechanism based on braking deceleration has been added, and the beep frequency will increase linearly with the braking deceleration to give drivers more intuitive early warning information.
	- **More comprehensive control**: All the functions and parameters that the whole algorithm depends on, such as the threshold of intersection ratio, confidence threshold, tracking enable, post-processing items of binocular camera, can be set through a unified configuration file.
	- **More portable and practical**: The algorithm is deployed to edge-side embedded hardware and consumes `11W` at full load. At the same time, frames and their processed information can be saved and played back in real time, which can completely replace traditional driving recorders.

## 📑 Instructions

1. Hardware

	|               `Jetson-TX2`               |         `Intel Realsense D435`          |
	|:----------------------------------------:|:--------------------------------------:|
	|             Architecture: `aarch64`              |        Firmware version: `05.13.00.50`         |
	| System: `Linux-Ubuntu 18.04 Bionic Beaver` | Resolution: `848 × 480 (max: 1920 × 1280)` |
	|                 Memory: 8G                 |          Frame rate: `30 (max: 90)`          |
	|       Graphics card: `(8G) NVIDIA Tegra X2`       |      Depth Field of View (FOV): `87° × 58°`       |
	|            `SoC`: `tegra186`             |      Minimum depth distance: `28 cm`      |
	|             `L4T`: `32.7.5`              |       RGB sensor FOV (H × V): `69° × 42°`       |
	|            `Jetpack`: `4.6.5`            |       RGB sensor technology: `Rolling Shutter`        |

2. Software
	- `CUDA`: `10.2.300`
	- `cuDNN`: `8.2.1.32`
	- `ROS`: `1.14.13`
	-  `Python`: `3.6.9`
	- `Pytorch`: `1.10.0`
	- `torchvision`: `0.11.1`
	- `OpenCV with CUDA`: `4.5.3`
	- `TensorRT`: `8.2.1.9`
	- `Pyrealsense2`: `2.50.0`
	- `onnx`: `1.11.0`
	- `onnxruntime`: `1.10.0`
	- `onnxslim`: `0.1.26`

3. Project structure
	- `Assets`: Store pictures, videos to be detected, and the font file that will be used in running.
	- `Dataset`: Store `TPZ` data label and data set production script files
	- `Main`: Store the project algorithm file
	- `Results`: Store the output file of the algorithm if specified `-s` argument (see below).
	- `Weights`: Store weight files,  as well as training script, weight- exporting script and weight-testing script.
	- `fast_run.sh`: A convenient script for algorithm running.
	- `button_run.sh`: A convenient script for algorithm running(used when you want to run program by pressing power button shortly)
	- `handle-powerbtn.sh`: The script when run on pressing power button shortly

## 🔨 Preparations

1. Download the project

	```shell
	# pwd: <any-where>
	git clone https://github.com/Ahzyuan/BS_thesis.git DOIC
	cd DOIC && git checkout DOIC
	```

2. Configure project dependencies
	1. Create a conda virtual environment: `conda create -n doic python=3.6.9`
	<font size=2, color=pink>(Note: restricted by `Jetpack`，python version can only be 3.6.9)</font>

	2. Install third-party libraries: `pip install -r requirements.txt`
	3. Install `pyrealsense2`: according to the instructions in [link](https://blog.csdn.net/Boris_LB/article/details/120750799)
		<font size=2, color=pink>

		(Note: 
		
		①. Pick proper version of `librealsense`, e.g. 2.50；

		②. Disconnect the camera from the `Jetson-TX2` during installation;
		
		③. Change the firmware version of `realsense-viewer` follow [link](https://dev.intelrealsense.com/docs/firmware-releases-d400#d400-series-firmware-downloads) 
		)
		</font>
	
	4. Install `tensorrt`: `tensorrt` is built-in on Jetson platform. Just execute the following command
	
		```shell
		# pwd: <any-where>
		ln -s /usr/lib/python3.6/dist-packages/tensorrt* \
		~/archiconda3/envs/doic/lib/python3.6/site-packages/
		```
	
	5. Install `ultralytics`: `ultralytics` acquires `Python>=3.8`，so following modification are required.
	
		```shell
		# pwd: <any-where>
		# conda activate doic 
		git clone https://github.com/ultralytics/ultralytics.git 
		
		cd ultralytics/hub 
		sed '92 a\ header = self.get_auth_header()' auth.py 
		sed -i '94s/:= self.get_auth_header():/:/' auth.py 
		
		cd ../utils 
		sed -i '4s/./_/' __init__.py 
		sed -i '47s/importlib.metadata/importlib_metadata/' __init__.py 
		sed '13c\ from importlib_metadata import metadata' checks.py 
	
		cd ../nn
		sed '288 a\             batch, _, *imgsz = bindings["images"].shape' autobackend.py
		
		cd ../engine
		sed -i '315a\
				try:
					self.args.imgsz = self.model.imgsz  # update imgsz
					self.args.batch = self.model.batch  # update batch
					self.imgsz = self.model.imgsz  # update imgsz
				except:
					try:
						import re
						pattern = re.compile('([0-9]+)x([0-9]+)')
						res = list(map(int, re.findall(pattern, model)[-1]))
						self.imgsz = getattr(self.args, 'imgsz', res)
					except:
						self.imgsz = getattr(self.args, 'imgsz', (384, 640))
				' predictor.py
		sed -i '559s/{6, 7}/{6, 7, 8, 9}/' results.py
	
		# create setup script 
		cd .. 
		cat >setup.py<EOF 
		from setuptools import setup, find_packages 
		
		setup(
			name="ultralytics", 
			version="8.2.59", 
			author='Ultralytics', 
			url='https://github.com/ultralytics/ultralytics', 
			description='YOLOV8', packages=find_packages(), 
		) 
		EOF 
		
		python setup.py install -e .
		```

3. Configure shortcut command

	```shell
	# pwd: <any-where>/DOIC
	
	echo alias doic="bash -i <any-where>/DOIC/fast_run.sh" >> ~/.bashrc
	source ~/.bashrc
	sudo chmod +x fast_run.sh
	```

4. Verification environment settings: If the above steps are completed successfully, you can execute the command `doic -h` on the terminal to get <font color=cyan>cyan text output</font>: 

	```plain-txt
	Activating environment ...
	Finish activation
	
	Running with args:
	> -h

	Loading Script in /home/yzq/hzy/DOIC/Main/main.py
	
	usage: main.py [-h] [-m MODEL] [-c CONFIG] [-i INPUT] [-s SAVE_DIR] [--show]
				   [--verbose]
	
	optional arguments:
	  -h, --help            show this help message and exit
	  -m MODEL, --model MODEL
							model path
	  -c CONFIG, --config CONFIG
							config file path
	  -i INPUT, --input INPUT
							input path, can be imgs, videos, directories, URLs or
							int for webcam
	  -s SAVE_DIR, --save_dir SAVE_DIR
							frame result save path
	  --show                show frame detection results on screen
	  --verbose             print frame detection results in terminal
	``` 

## 💡 Get Started

1. Model training

	> The weight that trained on the combination data of relabeled `D2City` and `D435` captured images with `imgsz=480` is provided! See [Weights/best_d2city_d435.pt](Weights/best_d2city_d435.pt) .
	>
	> If you want to train the model yourself, please follow the steps below.

	```shell
	# pwd: <any-where>/DOIC
	# conda activate doic

	# ---------------------- construct dataset ----------------------
	cd Dataset
	# Frame the original video according to the label.
	python framing.py -i <video path> -l <directory that label files in> -s <framed images storage directory>
	# Divide train set and val set randomly, default rate is 8:2
	python yolo_dataset_generator.py -i <framed images storage directory> -l <directory that label files in> -s <dataset storage path>

	# -------------------- prepare pretrained weight --------------------
	cd ../Weights
	wget https://bgithub.xyz/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt

	# ------------------------- start training ----------------------
	python train_da.py \
	-w yolov8s.pt \
	-t <target domain dataset path>/TPZ.yaml \
	-s <source domain dataset path>/TPZ.yaml \
	--imgsz 480 \
	--batch 16 \
	--enable_da \ # enable domain adptation
	--enable_train \ # enable training
	--val \ # eval model after each epoch
	--plots \
	--profile
	```
	
	> the training script [train_da.py](Weights/train_da.py) can be use in common training and domain adptation training.
	> 
	> Once specify `--enable_da`, it will enable domain adptation training which is based on `GRL(Gradient Reverse Layer)`, or it will just perform the common training.
	> 
	> If you want to enable domain adptation training,please make sure that the source domain dataset path and the target domain dataset path are different, or it is no different from common training.

2. Model transformation: transform the `pt` model to `onnx` and `tensorrt` format sequentially.

	```shell
	# pwd: <any-where>/DOIC
	cd Weights 
	
	# pt -> onnx(simplify)
	python export.py \ 
	-p 480s_best.pt \
	-s yolo8s_sim_288x480.onnx \ 
	-f o \ 
	--simplify 
	
	# add build-in tensorrt to ~/.bashrc
	export PATH=/usr/src/tensorrt/bin:$PATH >> ~/.bashrc
	source ~/.bashrc
	
	# The following operations to export the tensorrt model 
	# MUST be performed in jetson-tx2 !!!
	trtexec --onnx=yolo8s_sim_288x480.onnx \ 
	--saveEngine=yolo8s_half_sim_288x480.engine \ 
	--workspace=10240 \ # MB，the available memory for export must be >= 10 GB
	--fp16 # half precision inference, exclusive with `--half` argument in export.py
	--verbose # show detailed infomation when export
	```

2. Model testing: Execute the following command, and find the test results of all files under `Assets` in the `Results/demo` directory.

	```shell
	# pwd: <any-where>/DOIC/Weights
	python test_export.py \ 
	-m yolo8s_half_sim_288x480.engine \
	-s ../Assets \
	-d ../Results/demo
	```

3. `Jetson-TX2` connects `Realsense-435` binocular camera and speaker.
4. Configuration file settings: Set the content of the `DOIC/Main/TPZ.yaml` file, each meaning has been commented.
3. Start running: 

	```shell
	doic -m <tensorrt model path> -i 0 --show --verbose
	```

	**Arguments Description**: according to the output of `doic -h` in last section，the algorithm accepts following 6 arguments in total.
	- `--verbose`: If enabled, frame processing information will be displayed on the terminal in real time
	- `--show`: If enabled, the frame processing screen will be displayed on the screen in real time
	- `-m` / `--model`: `tensorrt` model path
	- `-c` / `--config`: Configuration file path
	- `-i` / `--input`: Input file path, accept images, videos, folders containing images and videos, and integer input. Among them, **when an integer is passed in, the camera input with the specified index is received.**
	- `-s` / `--save_dir`: Saving path of frame processing information, which is empty by default, indicating that frame processing information will not be saved. If specified, a folder will be created in the specified path according to the program startup time, and two subfolders, `img` and `meta`, will be established to store the frame image and its processing information respectively; If `verbose` argument is enabled, the terminal output information will be recorded in the `terminal_output.txt` file. An example of a output directory tree is as follows:

		```plain-txt
		.
		└── 2024-08-14T11-21-13
			├── img
			│   ├── image34.png
			│   ├── image35.png
			│   ├── image36.png
			│   ├── image37.png
			│   └── ...
			├── meta
			│   ├── image34.npy
			│   ├── image35.npy
			│   ├── image36.npy
			│   ├── image37.npy
			│   └── ...
			└── terminal_output.txt
		```

## 🚀 Advanced use: Control program running via power button

> If you want the program to run in any scenario in an offlined way without network connection, then this is what you might need.

1. Configure `acpid` to monitor power button short press event

	```shell
	# pwd: <any-where>/DOIC

	sudo apt-get update
	sudo apt-get install acpid

	sudo chmod +x handle-powerbtn.sh

	cat >/etc/acpi/events/powerbtn<< EOF
	event=button/power PBTN 00000080 00000000
	action=<any-where>/DOIC/handle-powerbtn.sh
	EOF

	sudo systemctl restart acpid
	```

2. Configure shortcut command

	```shell
	# pwd: <any-where>/DOIC
	
	echo alias doic_button="bash -i <any-where>/DOIC/button_run.sh" >> ~/.bashrc
	source ~/.bashrc
	sudo chmod +x button_run.sh
	```

3. Run `doic_button -m <tensorrt model path> -s <output path>` in terminal, the arguments are all the same with `doic`, but `-s` or `--save_dir` argument is needed this time.
4. Take the device to anywhere you want to go.
5. When the device gives a beep, press the power button briefly to start the program, and press the power button briefly again to end the program.

## 📝 License

This project follows the [Apache 2.0](LICENSE.sql) license terms. Without the permission of the project owner, the project can only be used for **academic and personal research** and cannot be used for any purpose that may be considered commercial.

