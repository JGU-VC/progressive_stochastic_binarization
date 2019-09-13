"""CIFAR-10 data set.

See http://www.cs.toronto.edu/~kriz/cifar.html.
"""
import os
import tensorflow as tf
from template import BaseDataSampler
import sys
import multiprocessing

HEIGHT = 224
WIDTH = 224
DEPTH = 3
datashape = [HEIGHT,WIDTH,DEPTH]
numexamples = 1281167
split = 0.999
DEFAULT_IMAGE_SIZE = 224
NUM_CHANNELS = 3
NUM_CLASSES = 1001



import data.imagenet.imagenet_preprocessing


class ImagenetDataset(object):
    """Cifar10 data set.

    Described by http://www.cs.toronto.edu/~kriz/cifar.html.
    """

    def __init__(self, data_dir, subset='train', use_distortion=True):
        self.data_dir = os.path.join(data_dir,"imagenet")
        self.subset = subset
        self.use_distortion = use_distortion

    def get_filenames(self):
        """Returns a python list of all (sharded) data subset files.

        Returns:
          python list of all (sharded) data set files.
        Raises:
          ValueError: if there are not data_files matching the subset.
        """
        tf_record_pattern = os.path.join(self.data_dir, '%s-*' % self.subset)
        data_files = tf.gfile.Glob(tf_record_pattern)
        if not data_files:
            print('No files found for dataset %s/%s at %s' % (self.name,
                self.subset,
                self.data_dir))

            self.download_message()
            sys.exit(-1)
        return data_files

    # def get_filenames(self):
    #     if self.subset == 'train':
    #         return [os.path.join(self.data_dir, 'train.tfrecords'), os.path.join(self.data_dir, 'validation.tfrecords')]
    #     if self.subset in ['train', 'validation', 'eval']:
    #         return [os.path.join(self.data_dir, self.subset + '.tfrecords')]
    #     else:
    #         raise ValueError('Invalid data subset "%s"' % self.subset)

    def parser(self, example_serialized, is_training):
        """Parses a single tf.Example into image and label tensors."""
        feature_map = {
              'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                                  default_value=''),
              'image/class/label': tf.FixedLenFeature([], dtype=tf.int64,
                                                      default_value=-1),
              'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                                     default_value=''),
        }
        sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
        # Sparse features in Example proto.
        feature_map.update({k: sparse_float32 for k in ['image/object/bbox/xmin',
                                           'image/object/bbox/ymin',
                                           'image/object/bbox/xmax',
                                           'image/object/bbox/ymax']})

        features = tf.parse_single_example(example_serialized, feature_map)
        label = tf.cast(features['image/class/label'], dtype=tf.int32)

        xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
        ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
        xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
        ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

        # Note that we impose an ordering of (y, x) just to make life difficult.
        bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

        # Force the variable number of bounding boxes into the shape
        # [1, num_boxes, coords].
        bbox = tf.expand_dims(bbox, 0)
        bbox = tf.transpose(bbox, [0, 2, 1])


        image_buffer = features["image/encoded"]
        image = imagenet_preprocessing.preprocess_image(
              image_buffer=image_buffer,
              bbox=bbox,
              output_height=DEFAULT_IMAGE_SIZE,
              output_width=DEFAULT_IMAGE_SIZE,
              num_channels=NUM_CHANNELS,
              is_training=is_training)
        image = tf.cast(image, tf.float32)

        return image, label

    def make(self, is_training):
        """Read the images and labels from 'filenames'."""
        filenames = self.get_filenames()
        # Repeat infinitely.
        dataset = tf.data.TFRecordDataset(filenames) #.repeat()

        # Parse records.
        dataset = dataset.map(lambda val: self.parser(val,is_training=is_training), num_parallel_calls=multiprocessing.cpu_count())
        # dataset = dataset.map(self.parser)

        return dataset


class DataSampler(BaseDataSampler):

    def num_classes(self):
        """Returns the number of classes in the data set."""
        return 1001 # +1 for unused background class

    def __init__(self, data_dir):
        self.data_dir = data_dir

        # download and extract
        # filepath = os.path.join(self.data_dir, "train.tfrecords")
        if not tf.gfile.Exists(self.data_dir):
            tf.gfile.MakeDirs(self.data_dir)
        # if not tf.gfile.Exists(filepath):
        #     print("No files found at "+filepath)

    def num_examples_per_epoch(self,subset='train'):
        if subset == 'train':
            return int(numexamples*(split))
        elif subset == 'validation':
            return int(numexamples*(1-split))
        elif subset == 'test':
            return 50000
        else:
            raise ValueError('Invalid data subset "%s"' % subset)

    def training(self):
        if not hasattr(DataSampler,"trainvaldata"):
            DataSampler.trainvaldata = ImagenetDataset(self.data_dir, subset='train', use_distortion=False).make(is_training=True)
        splitsize_train = self.num_examples_per_epoch("train")
        splitsize_test = self.num_examples_per_epoch("test")
        splitsize_val = self.num_examples_per_epoch("validation")
        print("Split ratio: ", splitsize_train," for Training, ", splitsize_val," for Validation,", splitsize_test," for Testing.")
        return DataSampler.trainvaldata.skip(splitsize_val)

    def validation(self):
        if not hasattr(DataSampler,"trainvaldata"):
            DataSampler.trainvaldata = ImagenetDataset(self.data_dir, subset='train', use_distortion=False).make(is_training=False)
        return DataSampler.trainvaldata.take(int(numexamples*(1-split)))

    def testing(self):
        return ImagenetDataset(self.data_dir, subset='validation', use_distortion=False).make(is_training=False)



