import os
import subprocess
import sys
import glob
import shutil
import urllib.request
import tarfile
import re
from time import sleep
from roboflow import Roboflow
from model_config import MODELS_CONFIG
from time import sleep

class TensorFlowObjectDetectionSetup:
    def __init__(self, base_path="content", selected_model='ssd_mobilenet_v2', num_train_steps=200, num_eval_steps=50):
        self.base_path = base_path
        os.chdir(self.base_path)

        self.selected_model = selected_model
        self.num_train_steps = num_train_steps
        self.num_eval_steps = num_eval_steps

        self.model_config = MODELS_CONFIG[self.selected_model]
        self.model_dir = 'training/'

        sys.path.append(os.path.join(base_path, 'models'))
        sys.path.append(os.path.join(base_path, 'models', 'research'))

        self.model_name = MODELS_CONFIG[self.selected_model]['model_name']
        self.pipeline_file = self.model_config['pipeline_file']
        self.pipeline_fname = self.pipeline_fname = os.path.join('models/research/object_detection/configs/tf2/', self.pipeline_file)
        self.batch_size = self.model_config['batch_size']
    
    # Download wildfire dataset
    def _download_dataset(self):
        rf = Roboflow(api_key="kurHDAP4tAun3nyyYi4d")
        project = rf.workspace("wildfire-xpwrf").project("wildfire-4tdl8")
        output_directory = "Wildfire"
        dataset = project.version(2).download("tfrecord", output_directory)
    
    # Download tensorflow repo, pick model parameters
    def _download_tensorflow_object_detection(self):
        repo_url = 'https://github.com/roboflow-ai/tensorflow-object-detection-faster-rcnn'
        repo_name = os.path.basename(repo_url)
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
    
    # Download model zoo and compile 
    def _download_model_repo(self):
        repo_url = 'https://github.com/tensorflow/models.git'
        #subprocess.run(['git', 'clone', '--quiet', repo_url], check=True)

        models_research_dir = os.path.join(self.base_path, 'models/research')
        protos_path = os.path.join(models_research_dir, 'object_detection', 'protos', '*.proto')
        for proto_file in glob.glob(protos_path):
            subprocess.run(['protoc', '--proto_path=' + models_research_dir, 
                            '--python_out=' + models_research_dir, proto_file], check=True)
    
    # Download a directory for the model
    def _download_training_dir(self):
        model_dir = 'training/'
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir, exist_ok=True)

    # Transfer dataset to model directory
    def _prep_train_test_data(self):
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
    
    # Install the selected model
    def _download_pretrained_model(self):
        model_name = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
        model_download_link = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/' 
        model_file = model_name + '.tar.gz'

        # Check if the model file already exists
        if not os.path.exists(model_file):
            print("Downloading model file:", model_file)
            urllib.request.urlretrieve(model_download_link + model_file, model_file)

        # Extract the model file
        print("Extracting model file:", model_file)
        with tarfile.open(os.path.join(os.getcwd(), model_file)) as tar:
            tar.extractall()

        # Clean up by removing the tar file
        os.remove(model_file)

        # Check if the destination directory exists, and if so, remove it
        if os.path.exists('models/research/pretrained_model'):
            shutil.rmtree('models/research/pretrained_model')

        # Rename the extracted folder to the destination directory
        os.rename(model_name, 'models/research/pretrained_model')
        print("Model is ready in directory:", 'models/research/pretrained_model')

    def download_repos(self):
        self._download_dataset()
        self._download_tensorflow_object_detection()
        self._download_model_repo()
        self._download_training_dir()
        self._prep_train_test_data()
        self._download_pretrained_model()  

    ## Configure training pipelines

    # Update checkpoint names
    def _fix_checkpoint_names(self):
        directory_path = 'models/research/pretrained_model/checkpoint'
        for filepath in glob.glob(os.path.join(directory_path, 'ckpt-0.*')):
            base_directory, filename = os.path.split(filepath)
            file_extension = filename.split('.', 2)[-1]
            
            new_filename = 'model.ckpt.' + file_extension
            new_filepath = os.path.join(base_directory, new_filename)
            os.rename(filepath, new_filepath)
            print(f'Renamed {filepath} to {new_filepath}')
        print(os.listdir(directory_path))

    def _configure_training_pipeline(self):
        assert os.path.isfile(self.pipeline_fname), '`{}` not exist'.format(self.pipeline_fname)
    
    def _get_num_classes(self, pbtxt_fname):
        from object_detection.utils import label_map_util
        label_map = label_map_util.load_labelmap(pbtxt_fname)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        num_classes = len(category_index.keys())
        return num_classes
    
    def _update_pipeline_config(self):
        label_map_pbtxt_fname = 'tensorflow-object-detection-faster-rcnn/data/train/fire_label_map.pbtxt'

        train_record = os.path.join(os.getcwd(), "tensorflow-object-detection-faster-rcnn/data/train/fire.tfrecord")
        train_label_map = os.path.join(os.getcwd(), "tensorflow-object-detection-faster-rcnn/data/train/fire_label_map.pbtxt")

        test_record = os.path.join(os.getcwd(), "tensorflow-object-detection-faster-rcnn/data/test/fire.tfrecord")
        test_label_map = os.path.join(os.getcwd(), "tensorflow-object-detection-faster-rcnn/data/test/fire_label_map.pbtxt")

        checkpoint = os.path.join(os.getcwd(), "models/research/pretrained_model/checkpoint/model.ckpt")
        num_classes = self._get_num_classes(label_map_pbtxt_fname)


        with open(self.pipeline_fname) as f:
            config_contents = f.read()
        
        # Update paths and settings within the configuration file
        config_contents = re.sub('fine_tune_checkpoint: ".*?"',
                                f'fine_tune_checkpoint: "{checkpoint}"', config_contents)
        config_contents = re.sub('fine_tune_checkpoint_type: ".*?"',
                                'fine_tune_checkpoint_type: "detection"', config_contents)
        config_contents = re.sub(r'(train_input_reader: {[\s\S]*?label_map_path: ")[^"]*(")',
                                r'\1{}\2'.format(train_label_map), config_contents)
        config_contents = re.sub(r'(eval_input_reader: {[\s\S]*?label_map_path: ")[^"]*(")',
                                r'\1{}\2'.format(test_label_map), config_contents)
        config_contents = re.sub(r'(train_input_reader: {\s+label_map_path: ".*?"\s+tf_record_input_reader {\s+input_path: ").*?(")',
                                r'\1{}\2'.format(train_record), config_contents, flags=re.DOTALL)
        config_contents = re.sub(r'(eval_input_reader: \{\s+label_map_path: ".*?"\s+shuffle: \w+\s+num_epochs: \d+\s+tf_record_input_reader \{\s+input_path: ").*?(")',
                                r'\1{}\2'.format(test_record), config_contents, flags=re.DOTALL)
        config_contents = re.sub('batch_size: [0-9]+',
                                f'batch_size: {self.batch_size}', config_contents)
        config_contents = re.sub('num_classes: [0-9]+',
                                f'num_classes: {num_classes}', config_contents)
        config_contents = re.sub('num_steps: [0-9]+',
                                f'num_steps: {self.num_train_steps}', config_contents)

        with open(self.pipeline_fname, 'w') as f:
            f.write(config_contents)
    
        print(config_contents)

    def training_pipeline(self):
        sys.path.append('/Users/divyaamirtharaj/Desktop/cs249_final_project/content/models')
        sys.path.append('/Users/divyaamirtharaj/Desktop/cs249_final_project/content/models/research')

        self._fix_checkpoint_names()
        self._configure_training_pipeline()
        self._update_pipeline_config()
    
    def run_training_script(self):
        self._update_pipeline_config()
        # Constructing the PYTHONPATH
        models_research_path = os.path.join(os.getcwd(), 'models', 'research')
        object_detection_path = os.path.join(models_research_path, 'object_detection')
        pythonpath = os.environ.get('PYTHONPATH', '')
        new_pythonpath = ':'.join([models_research_path, object_detection_path, pythonpath])

        model_dir = 'training/'

        # Construct the command
        command = [
            sys.executable, 
            "models/research/object_detection/model_main_tf2.py",
            f"--pipeline_config_path={self.pipeline_fname}",
            f"--model_dir={model_dir}",
            "--alsologtostderr",
            f"--num_train_steps={self.num_train_steps}",
            f"--num_eval_steps={self.num_eval_steps}"
        ]

        # Set up the environment for subprocess
        env = os.environ.copy()
        env['PYTHONPATH'] = new_pythonpath

        # Run the command
        subprocess.run(command, check=True, env=env)

setup = TensorFlowObjectDetectionSetup()
#setup.download_repos()
#setup.training_pipeline()
#setup.run_training_script()