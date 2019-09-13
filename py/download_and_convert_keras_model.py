import sys
import os

dataset = "imagenet"
print(sys.argv)
modelname = sys.argv[-2] if len(sys.argv)>2 else sys.argv[-1]
modelfile = sys.argv[-1] if len(sys.argv)>2 else None
destfile = os.path.join("ckpts_imgn",modelname+"_imagenet")
if os.path.exists(destfile+"/checkpoint"):
    print("model",modelname,"exists already. Nothing to do.")
    sys.exit(1)

# modelfile
if not modelfile:
    if modelname == "inceptioresnetv2":
        modelfile = "~/.keras/models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5"
    elif modelname == "mobilenet":
        modelfile = "~/.keras/models/mobilenet_1_0_224_tf.h5"
    elif modelname == "mobilenetv2":
        modelfile = "~/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5"
    elif modelname == "nasnetmobile":
        modelfile = "~/.keras/models/nasnet_mobile.h5"
    else:
        print("guessing modelfile from modelname")
if modelfile:
    modelfile = os.path.expanduser(modelfile)

# download graph
print("downloading model ",modelname)
import tensorflow as tf
from tensorflow import keras
import re
from classification_models import Classifiers
classifier, preprocess_input = Classifiers.get(modelname)
model = classifier((224, 224, 3), weights=dataset)

# actually builds the graph
data = tf.placeholder(shape=(None,224,224,3),dtype=tf.float32)
logits = model(data)

# guess the modelfile
modeldir = os.path.expanduser("~/.keras/models")
if modelfile is None:
    modelfile = [f for f in os.listdir(modeldir) if modelname+"_" in f] #ignores the version-suffixes
    if len(modelfile) == 0:
        modelfile = [f for f in os.listdir(modeldir) if re.sub(r"v\d+$","",modelname) in f] #ignores the version-suffixes
    print(modelfile)
    assert len(modelfile) == 1
    modelfile = os.path.join(modeldir,modelfile[0])

print("converting file:",modelfile, "for model ",modelname,"into file",destfile)


# Add ops to save and restore all the variables.
print("initialize variables in tensorflow session")
tf.train.get_or_create_global_step()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)

    print("reload weights into tensorflow session")
    model.load_weights(modelfile)
    os.makedirs(destfile,exist_ok=True)

    print("saving file")
    save_path = saver.save(sess, destfile+"/model.ckpt")

    # print("removing keras file")
    # os.remove(os.path.join(modeldir,modelfile))
