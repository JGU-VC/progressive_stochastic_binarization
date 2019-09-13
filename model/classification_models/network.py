import tensorflow as tf
import numpy as np
from template.misc import S, GLOBAL, get_distinct_colors, print_info, bcolors, reset_graph_uids
print = lambda *args: print_info(*args,color=bcolors.OKBLUE)

scopes_itself=True

from classification_models import Classifiers
from classification_models.keras_applications import keras
def network(data, labels_one_hot, mode):
    model_name = S("model.classification_models.model")
    dataset = S("model.classification_models.dataset")

    # keras.backend.set_learning_phase(1 if mode==tf.estimator.ModeKeys.TRAIN else 0) # 0: Test(default), 1: Train
    keras.backend.set_learning_phase(0) # 0: Test(default), 1: Train
    classifier, preprocess_input = Classifiers.get(model_name)

    # overwrite preprocess_input for mobilenet (workaround for a bug in keras_applications)
    if "mobilenet" in model_name:
        from keras.applications import imagenet_utils
        preprocess_input = lambda data: imagenet_utils.preprocess_input(data,mode='tf')

    # apply model
    data = preprocess_input(data)
    GLOBAL["keras_model_preprocess"] = preprocess_input
    model = classifier((224, 224, 3), input_tensor=data, weights=dataset)
    GLOBAL["keras_model"] = model
    logits = model.output

    # keras-models do not use empty-class
    logits = tf.concat([tf.expand_dims(logits[:,0]*0,1),logits],axis=-1)
    return logits
