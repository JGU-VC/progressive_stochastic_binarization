import tensorflow as tf
from tensorflow.python.ops.init_ops import Initializer
from tensorflow.python.util.tf_export import tf_export

tf_export("initializers.half")
class half(Initializer):
    def __init__(self, seed=None, dtype=tf.float32):
        self.seed = seed
        self.dtype = self.dtype

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        return 0.5*tf.ones(shape,seed=self.seed,dtype=dtype)

tf_export("initializers.transform")
class TransformInitializer(Initializer):
    def __init__(self, base_initializer, transformations, seed=None, dtype=tf.float32):
        self.transformations = transformations
        self.seed = seed
        self.dtype = dtype
        self.base_initializer = base_initializer

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        V = {"tf":tf}
        V["x"] = self.base_initializer(seed=self.seed)(shape,dtype=dtype)
        for t in self.transformations:
            V["x"] = eval(t,V)
        return V["x"]
