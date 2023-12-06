## Installation

Activate virtual environment:
```
source venv/bin/activate
```

Install requirements:
```
pip install -r requirements.txt
```

you might have to potentially download these separately sorry:
```
cd models/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install --use-feature=2020-resolver .
```

also you might have to do some work with versioning with tensorflow to get it to work (aka downloading 2.15.0 first, and then installing everything else, and then reinstalling and downgrading to 2.13.0)