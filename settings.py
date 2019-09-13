import tensorflow as tf
SETTINGS = {

    # model choice
    "model": {
        "type": "model.tf_resnet_official",
        # "type": "model.resnet",
        # "type": "model.small_conv",
        # "type": "model.slim",
    },

    "dataset": "data.imagenet",
    # "dataset": "data.mnist",
    # "dataset": "data.mnist_fashion",
    # "dataset": "data.cifar10",
    "dataset_mean_image_subtraction": "as True if S('dataset') and S('model.type') != 'model.classification_models' else False",
    "dataset_join_train_val": False,

    "data": {
        "dir": "download",
        # "format": "channels_first"
        "format": "channels_last" # needed for fold_batch_norms
    },

    # logging settings
    "log": {
        "dir": "as 'log' if S('model.type')!='model.classification_models' else 'ckpts_imgn/'+S('model.classification_models.model')+'_imagenet!'",

        "checkpoints": {
            "eachmins": 120,
            "restore": False,
        },

        "console": {
            "steps": "as S('log.console.real_steps')*S('optimizer.memory_size') if S('optimizer.use_custom') else S('log.console.real_steps')",
            "real_steps": 64
            # "real_steps": 16
            # "real_steps": 1
        },

        "summary": {
            "steps": "as S('log.console.steps')",
            "steps_validation": "as 25*S('log.console.steps')",
        },

        "summaries": True,

        "optimistic_restore": True
    },

    "debug": False,
    "log_device_placement": "as S('debug')",
    "train_only": False,
    "test_only": False,
    "test_mode": "test",
    "validation": "as False if S('dataset_join_train_val') else True",
    # "test_mode": "train",
    # "test_mode": "val",

    "test_subset": 1.0,


    # ================ #
    # predefined modes #
    # ================ #
    "preact_stats_first_batch": False,
    "preact_stats_by_image": False,
    "predict_patches": False,
    "attention_predict": False,
}


# settings for the pipeline
SETTINGS["batches"] = {

    # => batchs
    # "epoch": 90,
    "epoch": 35,
    # "epoch": 25,
    "repeat_each": "as S('optimizer.memory_size') if S('optimizer.use_custom') else 1",

    "size": 64,
    # "size": 256,

    "drop_remainder": False,
    # "in_buffer": 100, # 0: cache & shuffle full dataset, otherwise: # batches kept in cache
    "in_buffer": 10, # 0: cache & shuffle full dataset, otherwise: # batches kept in cache
    "prefetch": 1,

    # => one class per batch
    "same_class_at_same_position": False,
    "same_class_at_same_position_ordered": False,
    # "same_class_at_same_position": True,
    # "same_class_at_same_position_ordered": True,

    # => patches_size
    "patches_size": 0,
    # "patches_size": 28,

    # => testing #
    # "test_like_train": True,
    "test_like_train": False,
    "test_batch_size": 256,
    # "test_batch_size": "as S('batches.size')",

    "filter_class": -1
}



SETTINGS["optimizer"] = {


    # => learning rate
    "learning_rate": 5e-3,
    # "learning_rate": 1e-2,
    # "learning_rate": 1e-1, # cifar10
    # "learning_rate": 0.256, # cifar10

    "learning_rate_at": 0.3,

    # => epsilon
    "epsilon": 1.0, # imagenet
    # "epsilon": 1e-5,  # cifar

    # => custom optimizer
    # "use_custom": True,
    "use_custom": False,

    # => memory_size repeats batches, inferences net without training "memory_size-loss_collect_last" times
    "memory_size": 1,
    # "memory_size": 8,
    # "memory_size": 16,
    # "memory_size": 32,
    # "memory_size": 64,
    # "memory_size": 128,
    # "memory_size": 256,

    # => apply gradient to last x iterations per repetition
    # "collect_last": 255,
    # "collect_last": 10,
    "collect_last": 1,

    "decay": {
        # "type": "constant",

        "type": "exponential",
        "step": 10.0, # performs decay after step * NUMEXAMPLES steps
        # "step": 2.0, # performs decay after step * NUMEXAMPLES steps
        # "step": 100*64.0, # performs decay after step * NUMEXAMPLES steps
        # "rate": 0.94,
        "rate": 0.1,
        "staircase": True

        # "type": "superconvergence",

        # "type": "resnet",
    },

    "adam": {
        "beta1": 0.9,
        "beta2": 0.999,
    },

    "momentum": {
        "nesterov": True,
        "momentum": 0.9,
        # "momentum": 0.875,
    },

    "weight_decay": 5e-4,
    # "weight_decay": 1e-4,
    # "weight_decay": 0.00125,
}





# ---- #
# util #
# ---- #
SETTINGS["util"] = {

    # => custom library?
    # "tfl": "custom",
    # "tfl": "tf",
    "tfl": "tf_mod", # custom library by graph modification

    "variable": {

        # => base type
        "dtype": "float32",

        "name": "hiddenWeight",

        # seed
        "seed": None,

        # => initial value
        #
        # -> uniform distributions                #
        "initializer": "lecun_uniform",         # lecun normal, σ = sqrt(1 / fan_in)

        # => transformations (init, hidden, weight)
        # -> predefined
        "transformation": "real",  # (id, id, id)
        # "transformation": "2^k*p",  # (id, id, id)
        # "transformation": "2^k*binom_p",  # (id, id, id)
        # -> custom
        "transformation_templates": {

            "real": {
                "init": [],
                "hidden": [("p_init","p")],
                "weight": [
                    ("SUMMARY", 'tf.summary.histogram("w",w)'),
                ],
            },


            "psb": {
                "init": "as S('util.variable.transformation_templates.2^k*(1+m).init')",
                "hidden": "as S('util.variable.transformation_templates.2^k*(1+m).hidden')",
                "weight": "as S('util.variable.transformation_templates.2^k*(1+m).weight')",
            },

            "2^k*(1+m)": {
                "init": [],
                "hidden": [
                    ("p_init", 'p'),
                    ("SUMMARY", 'tf.summary.histogram("p",p)'),

                    # settings
                    ("ASSERT", "S('util.variable.scale.train')"),
                    ("min_2k", "S('util.variable.scale.min_2k')"),
                    ("significant_mode", "S('util.variable.significant.mode')"),
                    # ("sample_size", "S('binom.weight.adaptive_sample_size')[GLOBAL['weight_counter']]"),
                    # ("PRINT", "print('tensor',p)"),
                    ("sample_size", "S('binom.sample_size')"),

                    # move to probability
                    ("p_sign","tf.sign(p)"),
                    ("p_next_base2","tf.floor(tf.log(tf.maximum(tf.abs(p),min_2k))/tf.log(2.0))"),
                    ("p","tf.abs(p)/2**p_next_base2-1"), # p in [0,1)

                    # (fair) discretize / fixed_point-projection
                    "p if significant_mode=='real' else "
                        "fixed_point(p, S('util.variable.significant.bits')) if significant_mode=='fixed_point' else "
                        "quantize(p, S('util.variable.significant.bins'))",
                    "tf.identity(p,name=\"p\")",
                    ("SAVE", "GLOBAL[\"weights_p\"].update({p.name:p})"),

                    # summary
                    ("SUMMARY", 'tf.summary.histogram("p_m",p)'),
                    ("SUMMARY", 'tf.summary.histogram("p_2k",p_next_base2)'),
                ],
                "weight": [
                    # sample
                    ("m","sample_binomial(p,sample_size)/sample_size"),

                    # attention-mode: neuron
                    # only for --attention_predict
                    ("m", "m if not S('attention_predict') or S('attention.mode') != 'neuron' else "
                        "tf.where(tf.abs(p-0.5)<=S('attention.fraction'),"
                            "sample_binomial(p,S('attention.sample_size'))/S('attention.sample_size'),"
                            "m)"),
                    ("m_sum", "0 if not S('attention_predict') or S('attention.mode') != 'neuron' else "
                        "tf.reduce_sum(tf.where(tf.abs(p-0.5)<=S('attention.fraction'),"
                            "tf.ones_like(m),"
                            "tf.zeros_like(m)))"),
                    ("m_total", "tf.reduce_sum(tf.ones_like(m))"),
                    ("SAVE", "GLOBAL.update({'m_sum':m_sum+(GLOBAL[\"m_sum\"] if \"m_sum\" in GLOBAL else 0)})"),
                    ("SAVE", "GLOBAL.update({'m_total':m_total+(GLOBAL[\"m_total\"] if \"m_total\" in GLOBAL else 0)})"),

                    # redefine gradient
                    ("w", "pass_gradient(p_init,lambda p_init: p_sign*(2**p_next_base2*(1+m)))"),

                    # summaries
                    # "tf.identity(w,'the_weight')",
                    ("SUMMARY", 'tf.summary.histogram("binom",w)'),
                    # ("SUMMARY", 'tf.summary.histogram("w_add",w_add)'),
                ],
            },


        },

        # => regularizer
        "regularizer": {
            "type": None,
        },


        "significant": {
            "mode": "real",
            "bits": 64,

            # "mode": "fixed_point",
            # "bits": 4,
        },

        "scale": {
            "train": True,
            # "train": False,

            # "mode": "real",

            "mode": "2^k",
            "min_2k": 2**-16,
            # "mode": "fixed_point",
            # "bits": 2,
        },

        "globalscale": {
            "range": 2,
        },

        "fixed_point": {
            "max": 32,
            "bits": 16,
            "min": "as -S('util.variable.fixed_point.max')",
            "use": True,
            # "use": False,
        }
    }
}

SETTINGS["activation"] = {
    "type": "relu"
    # "type": "relu_mean_with_bias"
    # "type": "timerelu"
    # "type": "trsim"
}

SETTINGS["batch_norm"] = {
    # "transform": False,
    "transform": True,
}

# model dependent variables
SETTINGS["model"]["resnet"] = {
    "nettype": "tf_resnet",
    "first_conv_size": 3,
    # "first_conv_size": 3

    "num_conv_blocks": 8,

    "conv_blocks": "as S('model.resnet.num_conv_blocks')*[64]",
    # "conv_blocks": [64, 64],
    # "conv_blocks": [64]
    "memory_size": 16,
    # "memory_size": 64,

    # "last_layer_real": True,
    "last_layer_real": False,
}

# model dependent variables
SETTINGS["model"]["tf_resnet_official"] = {
    "version": 1,
    # "resnet_size": 50,
    "resnet_size": 18,
}

# model dependent variables
SETTINGS["model"]["classification_models"] = {
    "model": "resnet50",
    "dataset": "imagenet",
}

# model dependent variables
SETTINGS["model"]["small_conv"] = {
    "use_bias": True,
}



SETTINGS["pruning"] = {
    "activate": False,
    "sparsity": 0.99,
}



SETTINGS["binom"] = {
    # "gradient_correction": "none",
    "gradient_correction": "pass",
    # "gradient_correction": "gumbel",

    "probability_mode": "gumbel_log",
    # "probability_mode": "gumbel",
    # "probability_mode": "tfp",

    # "sample_size": 64,
    # "sample_size": 256,
    # "sample_size": 128,
    # "sample_size": 64,
    # "sample_size": 32,
    "sample_size": 16,
    # "sample_size": 8,
    # "sample_size": 2,
    # "sample_size": 1,
    # "sample_size": "ber",

    "weight": {
        "adaptive_sample_size": "as (S('model.resnet.num_conv_blocks')+1)*[S('binom.sample_size')]",
        # "adaptive_sample_size": [8,8,8,8,16,32,64,128],
        # "adaptive_sample_size": [16,16,32,32,64,64,128,128],
        # "adaptive_sample_size": [32,32,32,32,64,64,128,128],
        # "adaptive_sample_size": [4,4,8,8,16,16,32,32],
        # "adaptive_sample_size": [64,32,16,8,4,4,4,4],
    },

    "log_eps": 1e-8
}

SETTINGS["attention"] = {
    "mode": "spatial",
    "spatial_mode": "max_entropy",
    "spatial_surround": 1,
    "transform": "psb",
    "sample_size": 16,
    "fraction": 1.0,
}



# add git hash
import subprocess
def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
SETTINGS["log"]["git_hash"] = get_git_revision_short_hash().decode("utf-8")[:-1]

