import tensorflow as tf
import numpy as np
from util.variable import variableFromSettings
from template.misc import S, GLOBAL
from util.tfl import tfl

# identity
def id(x,num):
    return x

def relu(net):
    return tf.nn.relu(net)
