from ml.image.experiment.convnet_keras import ConvnetKerasImageExperiment
from ml.image.experiment.convnet_tf import ConvnetTensorflowImageExperiment

EXPERIMENT_REGISTRY = {
    ConvnetKerasImageExperiment.name:      ConvnetKerasImageExperiment,
    ConvnetTensorflowImageExperiment.name: ConvnetTensorflowImageExperiment
}
