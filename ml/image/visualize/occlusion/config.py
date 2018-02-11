import os

"""
MODEL_CLS_NAME = 'KerasModel'
MODEL_CLS_PATH = os.environ['MODEL_CLASS_PATH']
MODEL_LOAD_ARGS = {
    'model_path': os.environ['MODEL_PATH']
}
"""

MODEL_CLS_NAME = 'KerasModel'
MODEL_CLS_PATH = '/Users/jnewman/Projects/Banjo/ml/ml/visualize/occlusion/keras_model.py'
MODEL_LOAD_ARGS = {
    'class_map_path': '/Users/jnewman/Projects/Banjo/ml/ml/visualize/occlusion/class_map.json',
    'experiment_name': 'convnet_64x64x3_0.0.01_mse_adam',
    'model_path': '/Users/jnewman/Projects/Banjo/ml/model/fire_flood_v1/convnet_64x64x3_0.0.01_mse_adam_keras_1_2_chkpt.hdf5'
}
