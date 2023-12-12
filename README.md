## Model Approaches & Methodology

### Method 1: Roboflow
**Project Workspace:** https://universe.roboflow.com/wildfire-xpwrf/wildfire-4tdl8 <br><br>
+ ```Wildfire```: Generated TFRecords used in the TF pipeline
+ ```roboflow_yolov8_smokedetection.ipynb```: YOLOv8 notebook using Roboflow


### Method 2: MobileNet & Tensorflow
**Full Directory:** https://drive.google.com/drive/folders/1JGdd8-_YhyVWZcIfEldzXkmxJmdTw7t0?usp=sharing

+ ```model_config.py```: Configuration to get the relevant model zoo 
+ ```model_training.py```: Set up the full training pipeline for Tensorflow
+ ```inference.py```: Inference script 
+ ```job_config.slurm```: Job configuration to run on FAS RC

**Installation:**

Activate virtual environment (must be Python 3.10, and install basic requirements):
```
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Finish additional installation prior to running the script:
```
pip install tensorflow==2.15.0
pip install roboflow

pip install tf_slim
brew install protobuf
pip install -q Cython contextlib2 pillow lxml matplotlib
pip install -q pycocotools
pip install tensorflow_io
pip install tf-models-official --no-deps
```

Uncomment setup.download_repos() and run.  Then once completed, recomment it out and run setup.training_pipeline().  If this fails, proceed with the next steps, and then retry:
```
cd content/models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install . --no-deps
```

Recomment out setup.training_pipeline() and run the following:
```
pip install tensorflow==2.13.0
pip install gin
pip install lvis
pip install gin-config
```

Uncomment setup.run_training_script() to train.<br><br>


### Method 3: YOLOv7 & Pytorch
**Full Directory:** https://drive.google.com/drive/folders/1yn5tAXYjLVGM7h3v78Bj1WopDT_uTs5S?usp=sharing
+ ```yolov7_detect.py```: Generates the bounding boxes
+ ```yolov7_train.py```: Trains the model
+ ```yolov7_test```: Tests the model
+ ```yolov7_quantize.py```: Quantization for YOLOv7 model 
