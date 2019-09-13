import sys
import tensorflow as tf
from tensorflow import keras
import os

modelname = sys.argv[-2]
filename = sys.argv[-1]
print("converting file:",filename, " for model ",modelname)


from classification_models import Classifiers
classifier, preprocess_input = Classifiers.get(modelname)
model = classifier((224, 224, 3), weights='imagenet')
data = tf.placeholder(shape=(None,224,224,3),dtype=tf.float32)
logits = model(data)

# Add ops to save and restore all the variables.
# model = keras.models.load_model_weights(filename)
tf.train.get_or_create_global_step()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
saver = tf.train.Saver()
with tf.Session() as sess:
    # keras.set_session(sess)
    # sess = keras.backend.get_session()
    sess.run(init_op)
    print("load weights")
    model.load_weights(filename)
    print(filename[:-3])
    os.makedirs(filename[:-3]+"_tf",exist_ok=True)
    save_path = saver.save(sess, filename[:-3]+"_tf/model.ckpt")
