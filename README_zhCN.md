# 基于多目标检测与跟踪的车辆自动礼让行人算法研究与应用

![Static Badge](https://img.shields.io/badge/License-Apache_2.0-red?link=https://www.apache.org/licenses/LICENSE-2.0) ![Static Badge](https://img.shields.io/badge/SCAU-DOIC_2024%3A_%E6%95%B0%E6%8D%AE%E5%BC%80%E6%94%BE%E5%BA%94%E7%94%A8%E5%A4%A7%E8%B5%9B-green?labelColor=007A49&link=https%3A%2F%2Fmetc.scau.edu.cn%2Fscaudoic2024%2F)

[![Static Badge](https://img.shields.io/badge/Ultralytics-YOLOv8.2-purple)](https://github.com/ultralytics/ultralytics) ![Static Badge](https://img.shields.io/badge/TensorRT-JetsonTX2-green?logo=nvidia&color=76B900&link=https%3A%2F%2Fdeveloper.nvidia.cn%2Ftensorrt&link=https%3A%2F%2Fdeveloper.nvidia.com%2Fembedded%2Fjetson-tx2) ![Static Badge](https://img.shields.io/badge/Intel-RealSense--D435-lightcyan?logo=intel&color=F4F4F4&link=https%3A%2F%2Fwww.intelrealsense.com%2Fdepth-camera-d435%2F)

---
英文版：[README](README.md)

## 😀 项目介绍

1. 本项目在 [YOLOv8](https://github.com/ultralytics/ultralytics) 的基础上，依托双目深度相机 `RealSense-D435` ，通过 `Jetson-TX2 (arm64-ubuntu18.04)` 平台，改进了主分支所提出的算法，**提高了其在稳定性、准确性与实时性方面的表现，并实现了其在边缘端的部署。**

2. 本项目借鉴了 `YOLOv8` 的实现流程与编码风格，通过对 `YOLOv8` 与主分支所提出的算法的修改，使本项目能与原 `YOLOv8` 相容的同时，在实现上增强了模块的封装性与易用性，在功能上增强了结果稳定性与准确性。

3. 具体来说，本项目所具有的功能有：
	- **原功能 100% 兼容**：依旧可实现行车记录仪视角下的车辆自动礼让行人决策
	- **精确测距**：兼容原有的视频、图片输入的同时，可开启双目相机，实现更精准的视觉测距。同时实现了双目相机的多种高级设置接口，如自动曝光、自动白平衡、深度空洞填充，确保测距值对环境变化的鲁棒性。
	- **精准跟踪**：兼容原有的基于规则的目标跟踪算法的同时，可根据硬件资源启用多种的专业跟踪算法，如 [ByteTrack](https://github.com/ifzhang/ByteTrack)、[BoTSORT](https://github.com/viplix3/BoTSORT-cpp)
	- **准确性与实时性的更高平衡**：通过 `pt -> onnx -> tensorRT` 的模型转换路径，充分发挥 `tensorRT` 的层融合机制完成模型的提纯。利用 `yolov8s.pt` 所转换的模型，在 `Jetson-TX2` 上运行原有算法 + 双目测距 + `ByteTrack` 跟踪算法，帧率依旧可稳定在 `28~32 fps`。
	- **更直观的交互**：通过 `rich` 库实现更多样、更丰富的终端信息反馈。同时，包括深度-彩色图像融合、决策目标分类、灯色判断等在内的帧处理信息，更易获取、显示与保存。另外，新增了基于制动减速度的蜂鸣报警机制，蜂鸣频率将随着制动减速度线性增长，以给予驾驶员更直观的预警信息。
	- **更全面的运行控制**：整个算法运行依赖的各项功能与超参，如推理时交并比阈值、置信度阈值、跟踪启用、双目相机后处理项等，全部可通过一个统一的配置文件进行设置。
	- **更便携与实用**：算法部署到边缘端嵌入式硬件，满载功耗 `11W`。同时，帧及其处理信息可实时保存与回放，完全可替代传统的行车记录仪。

## 📑 项目说明

1. 硬件条件

	|               `Jetson-TX2`               |         `Intel Realsense D435`          |
	|:----------------------------------------:|:--------------------------------------:|
	|             架构：`aarch64`              |        固件版本：`05.13.00.50`         |
	| 系统：`Linux-Ubuntu 18.04 Bionic Beaver` | 分辨率：`848 × 480 (最大 1920 × 1280)` |
	|                 内存：8G                 |          帧率：`30 (最大 90)`          |
	|       显卡：`(8G) NVIDIA Tegra X2`       |      双目相机视场角：`87° × 58°`       |
	|            `SoC`：`tegra186`             |      双目相机最近测距值：`28 cm`      |
	|             `L4T`：`32.7.5`              |       RGB相机视场角：`69° × 42°`       |
	|            `Jetpack`：`4.6.5`            |       RGB相机快门类型：卷帘快门        |

2. 软件版本
	- `CUDA`：`10.2.300`
	- `cuDNN`：`8.2.1.32`
	- `ROS`：`1.14.13`
	-  `Python`：`3.6.9`
	- `Pytorch`：`1.10.0`
	- `torchvision`：`0.11.1`
	- `OpenCV with CUDA`：`4.5.3`
	- `TensorRT`：`8.2.1.9`
	- `Pyrealsense2`：`2.50.0`
	- `onnx`：`1.11.0`
	- `onnxruntime`：`1.10.0`
	- `onnxslim`：`0.1.26`
3. 项目目录
	- `Assets`：存放待检测的图片、视频等
	- `Dataset`：存放 `TPZ` 数据标签与数据集制作等脚本文件
	- `Main`：存放项目算法文件
	- `Results`：存放算法运行输出文件
	- `Weights`：存放各种格式的权重文件与权重导出、测试脚本
	- `fast_run.sh`：算法便捷运行脚本

## 🔨 项目部署

1. 下载项目

	```shell
	# pwd: <any-where>
	git clone https://github.com/Ahzyuan/BS_thesis.git DOIC
	cd DOIC && git checkout DOIC
	```

2. 配置项目依赖
	1. 创建虚拟环境：`conda create -n doic python=3.6.9`
	<font size=2, color=pink>(注：`Jetpack` 限制，只能 3.6.9)</font>

	2. 安装第三方库：`pip install -r requirements.txt`
	3. 安装 `pyrealsense2`：遵循[链接]([https://blog.csdn.net/Boris_LB/article/details/120750799](https://blog.csdn.net/Boris_LB/article/details/120750799))指示
	<font size=2, color=pink>
	
		(注意：

		①. 选择合适的 `librealsense` 版本, 如 2.50;

		②. 安装时需断开相机与 `Jetson-TX2` 的连接;

		③. 需要根据[链接](https://dev.intelrealsense.com/docs/firmware-releases-d400#d400-series-firmware-downloads)更改 `realsense-viewer` 的固件版本 )
	</font>
	
	4. 安装 `tensorrt`：`tensorrt` 为 `Jetson` 平台内置，执行以下命令即可：
	
		```shell
		# pwd: <any-where>
		ln -s /usr/lib/python3.6/dist-packages/tensorrt* \
		~/archiconda3/envs/doic/lib/python3.6/site-packages/
		```
	
	5. 安装 `ultralytics`：`ultralytics` 原本要求 `Python>=3.8`，需进行以下改动
	
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

3. 配置快捷运行命令

	```shell
	# pwd: <any-where>/DOIC
	# 注：替换下方的<DOIC_dir>为项目所在的绝对路径
	echo alias doic="bash -i <DOIC_dir>/fast_run.sh" >> ~/.bashrc
	source ~/.bashrc
	sudo chmod +x fast_run.sh
	```

4. 检验环境设置：若上述步骤顺利完成，可在终端执行命令 `doic -h`，可得<font color=cyan>青色文本输出</font>：

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

## 💡 项目使用

1. 模型转换

	```shell
	# pwd: <any-where>/DOIC
	cd Weights 
	
	# pt -> onnx(simplify)
	python export.py \ 
	-p 480s_best.pt \
	-s yolo8s.onnx \ 
	-f o \ 
	--simplify 
	
	# 将内置的 tensorrt 加入系统环境变量
	export PATH=/usr/src/tensorrt/bin:$PATH >> ~/.bashrc
	source ~/.bashrc
	
	# 以下导出 tensorrt 模型的操作，必须在 jetson-tx2 中进行
	trtexec --onnx=yolo8s_sim_288x480.onnx \ 
	--saveEngine=yolo8s_half_sim_288x480.engine \ 
	--workspace=10240 \ # MB，导出时可用的内存，保证 >=10G
	--fp16 # 半精度推理, 与 export.py 中的参数 `--half` 相斥
	--verbose # 显示详细导出信息
	```

2. 模型测试：执行以下命令，可在 `Results/demo` 目录找到 `Assets` 下所有文件的检测结果

	```shell
	# pwd: <any-where>/DOIC/Weights
	python test_export.py \ 
	-m yolo8s_half_sim_288x480.engine \
	-s ../Assets \
	-d ../Results/demo
	```

3. `Jetson-TX2` 连接 `Realsense-435` 双目相机与扬声器
4. 配置文件设置：设置 `DOIC/Main/TPZ.yaml` 文件内容，各项含义已注释
3. 运行算法：根据上一节 `doic -h` 的输出，算法共接受以下6个运行参数
	- `--verbose`：帧处理信息实时显示在终端
	- `--show`：帧处理画面实时显示在屏幕
	- `-m` / `--model`：`tensorrt` 模型路径
	- `-c` / `--config`：配置文件路径
	- `-i` / `--input`：输入文件路径，接受图片、视频、包含图片与视频的文件夹以及整数输入。其中，**传入整数则接收索引为指定整数的摄像机输入。**
	- `-s` / `--save_dir`：帧处理信息保存路径，默认为空，表示不保存帧处理信息。若指定路径，则将在指定目录按程序启动时间创建文件夹，并建立 `img` 与 `meta` 两个子文件夹，分别存储帧图像及其处理信息；同时，终端输出信息会记录至 `terminal_output.txt` 文件。一个目录树示例如下：

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

## 📝 许可声明

本项目遵循 [Apache 2.0 许可条款](LICENSE.sql)。若无项目作者允许，该项目仅可用于**学术与个人研究使用**，而不可用于任何可能被视为商业用途的用途。