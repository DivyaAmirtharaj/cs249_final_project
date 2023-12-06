from roboflow import Roboflow
import os
import subprocess
import sys
from model_config import MODELS_CONFIG
from time import sleep
import glob
import shutil
import urllib.request
import tarfile

# Global base path
BASE_PATH = "content"
os.chdir(BASE_PATH)

# Install Roboflow dataset
def install_dataset():
    rf = Roboflow(api_key="kurHDAP4tAun3nyyYi4d")
    project = rf.workspace("wildfire-xpwrf").project("wildfire-4tdl8")
    output_directory = "Wildfire"
    dataset = project.version(2).download("tfrecord", output_directory)

# Download tensorflow repo, pick model parameters
def setup_tensorflow_object_detection(num_steps=1000, num_eval_steps=50, selected_model='ssd_mobilenet_v2'):
    """
    Args:
    num_steps (int): Number of training steps.
    num_eval_steps (int): Number of evaluation steps.
    selected_model (str): Model to use from predefined configurations.
    """
    
    repo_url = 'https://github.com/roboflow-ai/tensorflow-object-detection-faster-rcnn'
    repo_name = os.path.basename(repo_url)
    print(repo_name)

    # Ensure the BASE_PATH directory exists
    repo_dir_path = repo_name

    # Clone the repository if it doesn't exist
    if not os.path.exists(repo_dir_path):
        try:
            subprocess.run(["git", "clone", repo_url], check=True)
        except subprocess.CalledProcessError:
            print("Failed to clone the repository.")
            sys.exit(1)
    else:
        print(f"Repository already exists at {repo_dir_path}")

    # Retrieve model configuration
    MODEL = MODELS_CONFIG[selected_model]['model_name']
    pipeline_file = MODELS_CONFIG[selected_model]['pipeline_file']
    batch_size = MODELS_CONFIG[selected_model]['batch_size']

    return MODEL, pipeline_file, batch_size

# Install model zoo and compile 
def install_model_directory():
    repo_url = 'https://github.com/tensorflow/models.git'
    subprocess.run(['git', 'clone', '--quiet', repo_url], check=True)

    models_research_dir = os.path.join(os.getcwd(), 'models/research')
    object_detection_protos_dir = os.path.join(models_research_dir, 'object_detection', 'protos')

    # Find all .proto files in object_detection/protos and compile them
    proto_files = glob.glob(os.path.join(object_detection_protos_dir, '*.proto'))
    for proto_file in proto_files:
        subprocess.run(['protoc', '--proto_path=' + object_detection_protos_dir, '--python_out=' + models_research_dir, proto_file], check=True)

def prepare_tfrecord_files():
    source_dir = os.path.join('Wildfire')
    target_dir = os.path.join('tensorflow-object-detection-faster-rcnn', 'data')

    # Copy training and test data
    for dataset_type in ['train', 'test']:
        source_dataset_dir = os.path.join(source_dir, dataset_type)
        target_dataset_dir = os.path.join(target_dir, dataset_type)
        if os.path.exists(source_dataset_dir):
            shutil.copytree(source_dataset_dir, target_dataset_dir)
        else:
            print(f"Dataset directory {source_dataset_dir} does not exist.")

def download_pretrained_model(model_name, model_download_link):
    MODEL_FILE = MODEL + '.tar.gz'

    # Check if the model file already exists
    if not os.path.exists(MODEL_FILE):
        print("Downloading model file:", MODEL_FILE)
        urllib.request.urlretrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)

    # Extract the model file
    print("Extracting model file:", MODEL_FILE)
    with tarfile.open(os.path.join(os.getcwd(), MODEL_FILE)) as tar:
        tar.extractall()

    # Clean up by removing the tar file
    os.remove(MODEL_FILE)

    # Check if the destination directory exists, and if so, remove it
    if os.path.exists('models/research/pretrained_model'):
        shutil.rmtree('models/research/pretrained_model')

    # Rename the extracted folder to the destination directory
    os.rename(MODEL, 'models/research/pretrained_model')
    print("Model is ready in directory:", 'models/research/pretrained_model')

MODEL = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/'

download_pretrained_model(MODEL, DOWNLOAD_BASE)