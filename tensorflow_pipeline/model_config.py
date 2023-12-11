MODELS_CONFIG = {
    'ssd_mobilenet_v2': {
        'model_name': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8',
        'pipeline_file': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config',
        'batch_size': 12
    },
    'faster_rcnn': {
        'model_name': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu',
        'pipeline_file': 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.config',
        'batch_size': 12
    },
    'efficientdet': {
        'model_name': 'ssd_efficientdet_d4_1024x1024_coco17_tpu-32',
        'pipeline_file': 'ssd_efficientdet_d4_1024x1024_coco17_tpu-32.config',
        'batch_size': 8
    },
}