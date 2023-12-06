from roboflow import Roboflow
import os
import subprocess
import sys
from model_config import MODELS_CONFIG
from time import sleep
import glob

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

# Install model zoo and 
def install_model_directory():
    repo_url = 'https://github.com/tensorflow/models.git'
    subprocess.run(['git', 'clone', '--quiet', repo_url], check=True)

    models_research_dir = os.path.join(os.getcwd(), 'models/research')
    object_detection_protos_dir = os.path.join(models_research_dir, 'object_detection', 'protos')

    # Find all .proto files in object_detection/protos and compile them
    proto_files = glob.glob(os.path.join(object_detection_protos_dir, '*.proto'))
    for proto_file in proto_files:
        subprocess.run(['protoc', '--proto_path=' + object_detection_protos_dir, '--python_out=' + models_research_dir, proto_file], check=True)


