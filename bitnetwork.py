print_orig = print
from template.misc import IteratorInitializerHook, CustomSummarySaverHook, OnceSummarySaverHook, print_and_override_settings, settings_add_to_argparse, S, reset_graph_uids, print_info as print, bcolors
from util.helpers import get_tensor, learning_rate_with_decay, optimistic_restore
from util import fold_batch_norms
import os
import util.loops as loops
import sys

import tensorflow as tf
import numpy as np
import time
from settings import SETTINGS



if __name__ == '__main__':

    try:

        # ------------- #
        # cli arguments #
        # ------------- #

        import argparse
        import importlib

        # enable tf logging, show DEBUG output
        tf.logging.set_verbosity(tf.logging.DEBUG)

        # create cli arguments based on settings
        parser = argparse.ArgumentParser()
        settings_add_to_argparse(SETTINGS,parser)
        args = parser.parse_args()

        # print & save arguments
        print_and_override_settings(SETTINGS,args)

        # GLOBAL may depend on SETTINGS
        from template.misc import GLOBAL

        # import data
        dataset = importlib.import_module(S("dataset"))
        sampler = GLOBAL["dataset"] = dataset.DataSampler(S("data.dir"))
        STEPS_PER_EPOCH = 0

        # import model
        model = importlib.import_module(S("model.type"))
        losses = importlib.import_module('.'.join([S("model.type"), "loss"]))
        networks = importlib.import_module('.'.join([S("model.type"), "network"]))
        # define network and loss function
        network = networks.network
        lossfn = losses.lossfn

        REPEAT_BATCH = S("batches.repeat_each")
        EPOCHS = S("batches.epoch")



        # ----------------- #
        # define train data #
        # ----------------- #

        def define_data(ds, numexamples, mode):
            batch_size = S("batches.size")
            num_classes = GLOBAL["dataset"].num_classes()
            steps_per_epoch = numexamples/batch_size

            if mode == tf.estimator.ModeKeys.TRAIN:
                # make buffered shuffle and repeat for each epoch
                if S("batches.in_buffer") == 0:
                    ds = ds.shuffle(numexamples,reshuffle_each_iteration=True)
                else:
                    ds = ds.shuffle(batch_size*S("batches.in_buffer"),reshuffle_each_iteration=True)
                ds = ds.repeat(EPOCHS)
            elif mode == tf.estimator.ModeKeys.EVAL:
                pass
            else:
                raise ValueError("Mode Key not known")


            # if S("batches.filter_class") >= 0:
            #     ds = ds.filter(lambda data, label: tf.equal(label, S("batches.filter_class")))

            # one class per batch (optional)
            if S("batches.same_class_at_same_position"):
                ds_class = []
                for i in  range(num_classes):
                    ds_c = ds.filter(lambda data, label: tf.equal(label, i))
                    ds_class.append(ds_c)
                ds = tf.data.Dataset.zip(tuple(ds_class))

                def shuffle_classes(*classes):
                    if S("batches.same_class_at_same_position_ordered"):
                        permutation = np.arange(num_classes)
                    else:
                        permutation = np.random.permutation(num_classes)
                    data = tf.data.Dataset.from_tensors(classes[permutation[0]])
                    for i in range(1,num_classes):
                        data = data.concatenate(tf.data.Dataset.from_tensors(classes[permutation[i]]))
                    return data
                ds = ds.flat_map(shuffle_classes)
                batch_size *= num_classes

            # make batches
            ds = ds.batch(batch_size, drop_remainder=S("batches.drop_remainder"))

            # repeat each batch
            if REPEAT_BATCH > 1:
                ds = ds.flat_map(lambda x,y: tf.data.Dataset.from_tensors((x,y)).repeat(REPEAT_BATCH))
                # ds = ds.flat_map(lambda x,y: tf.data.Dataset.from_tensors((tf.concat([x]*64, axis=0),tf.concat([y]*64, axis=0))).repeat(REPEAT_BATCH))
                steps_per_epoch *= REPEAT_BATCH

            # patches_size
            prefetch_mult = 1
            if S("batches.patches_size") and S("batches.patches_size") > 0:
                def patches_size(x,y):
                    shape = x.shape.as_list()
                    size = int(S("batches.patches_size"))

                    width, height = shape[2], shape[1]
                    width_num, height_num = width//size, height//size
                    print("splitting batches to patches of size :", width_num, height_num)
                    patches = []
                    GLOBAL["patches"] = 0

                    # divide image into patches
                    patches = tf.split(tf.concat(tf.split(x,width_num,axis=1),0),height_num,axis=2)
                    GLOBAL["patches"] += width_num*height_num

                    # ... also do overlap
                    if width_num != 1 and height_num != 1:
                        x_overlap = x[:,size//2:-size//2,size//2:-size//2,:]
                        patches += tf.split(tf.concat(tf.split(x_overlap,width_num-1,axis=1),0),height_num-1,axis=2)
                        GLOBAL["patches"] += (width_num-1)*(height_num-1)

                    # concats to: id0_p0, id1_p0, id2_p0, ...
                    patches = tf.concat(patches,0)
                    labels = tf.concat([y]*GLOBAL["patches"], axis=0)
                    return (patches, labels)
                ds = ds.map(patches_size)
                prefetch_mult *= GLOBAL["patches"]
                # ds = ds.flat_map(lambda x,y: tf.data.Dataset.from_tensors(patches_size(x, y)))
                print("Space To Batch-size results in:",ds)

            ds = ds.prefetch(S("batches.prefetch")*prefetch_mult)
            return ds, steps_per_epoch

        # define training dataset
        train_ds = sampler.training()
        train_ds, STEPS_PER_EPOCH = define_data(train_ds, sampler.num_examples_per_epoch("train"), tf.estimator.ModeKeys.TRAIN)
        STEPS_TOTAL = STEPS_PER_EPOCH*EPOCHS


        print()
        print()
        print("\tNUMEXAMPLES:",sampler.num_examples_per_epoch("train"))
        print("\tSTEPS_PER_EPOCH:",STEPS_PER_EPOCH)
        print()
        print()


        # ---------------------- #
        # define validation data #
        # ---------------------- #

        if S("validation"):

            # define validation dataset
            validation_ds = sampler.validation()
            validation_ds = validation_ds.take(S('batches.size')).repeat()
            if S("batches.test_like_train"):
                validation_ds, _ = define_data(validation_ds, sampler.num_examples_per_epoch("validation"), tf.estimator.ModeKeys.EVAL)
            else:
                validation_ds = validation_ds.batch(256)


        # ---------------- #
        # define test data #
        # ---------------- #

        # define test dataset
        if "test" in S("test_mode"):
            test_ds = sampler.testing()
        elif "train" in S("test_mode"):
            test_ds = sampler.training()
        elif "val" in S("test_mode"):
            test_ds = sampler.validation()
        else:
            raise ValueError("Test-mode not known")
        test_size = sampler.num_examples_per_epoch(S("test_mode"))
        print("test-size (subset="+S("test_mode")+"):",test_size)
        if S("test_subset") != 1.0: # for debugging purposes
            test_size = int(test_size*S("test_subset"))
            test_ds = test_ds.take(test_size)
            print(bcolors.WARNING+"    using subset of testset of size:",test_size)

        # make batches
        if S("preact_stats_first_batch"):
            batch_id = 0
            # batch_id = 3*64+57
            # batch_id = 0*S("batches.size")
            # batch_id = 1*S("batches.size")
            # batch_id = 1*S("batches.size")+12
            batch_id = 1*32+12

            if S("batches.filter_class") >= 0:
                test_ds = test_ds.filter(lambda data, label: tf.equal(label, S("batches.filter_class")))
            test_ds = test_ds.skip(batch_id).take(S("batches.size")).repeat().batch(S("batches.size"))
        else:
            if S("batches.test_like_train"):
                test_ds, _ = define_data(test_ds, sampler.num_examples_per_epoch("test"), tf.estimator.ModeKeys.EVAL)
                if S("batches.patches_size") and S("batches.patches_size") > 0:
                    test_size *= S("batches.patches_size")**2
            else:
                # TODO: remove
                # batch_id = 1*32+12
                # batch_id = 1
                # batch_id = (1*32+12)//S("batches.size")
                # if S("batches.filter_class") >= 0:
                #     test_ds = test_ds.filter(lambda data, label: tf.equal(label, S("batches.filter_class")))
                # train_ds = train_ds.take(S("batches.size")).repeat()
                # test_ds = test_ds.batch(S("batches.size")).skip(batch_id).take(1).repeat()
                # test_size = S("batches.size")

                test_ds = test_ds.batch(S("batches.test_batch_size"))


        # --------------------- #
        # post dataset-creation #
        # --------------------- #

        if S("batches.same_class_at_same_position"):
            SETTINGS["batches"]["size"] *= GLOBAL["dataset"].num_classes()
            SETTINGS["batches"]["in_buffer"] *= GLOBAL["dataset"].num_classes()



        # ================ #
        # initialize graph #
        # ================ #

        # A reinitializable iterator is defined by its structure. (same graph for train and testset)
        with tf.name_scope("data"):
            if not S("test_only"):
                iterator = tf.data.Iterator.from_structure(train_ds.output_types, train_ds.output_shapes)
            else:
                iterator = tf.data.Iterator.from_structure(test_ds.output_types, test_ds.output_shapes)
            with tf.name_scope("train"):
                train_ds_it_init = iterator.make_initializer(train_ds)
            with tf.name_scope("test"):
                test_ds_it_init = iterator.make_initializer(test_ds)
            with tf.name_scope("validation"):
                if S("validation"):
                    validation_data = validation_ds.make_one_shot_iterator().get_next()
            data = iterator.get_next()

        # learning rate
        current_epoch = tf.cast(tf.train.get_or_create_global_step(),tf.float32)/STEPS_PER_EPOCH
        learning_epsilon = S("optimizer.epsilon")
        lr_base = S("optimizer.learning_rate")
        lr_base_at = S("optimizer.learning_rate_at")*EPOCHS
        if S("optimizer.decay.type") == "constant":
            learning_rate = tf.constant()
        elif S("optimizer.decay.type") == "exponential":
            learning_rate = tf.train.exponential_decay(
                    learning_rate=S("optimizer.learning_rate"),              # Base learning rate.
                    global_step=tf.train.get_or_create_global_step(),      # Current index into the dataset.
                    decay_steps=int(S("optimizer.decay.step")*sampler.num_examples_per_epoch("train")/S("batches.size")),     # Decay step.
                    # decay_steps=S("optimizer.decay.step"),
                    decay_rate=S("optimizer.decay.rate"),                 # Decay rate.
                    staircase=S("optimizer.decay.staircase"),
                    name="learning_rate")
        elif S("optimizer.decay.type") == "superconvergence":
            lr_small = 0.25*lr_base
            lr_small_at = 0.6*EPOCHS

            learning_rate = tf.where(current_epoch <= lr_base_at,
                    lr_base*current_epoch/lr_base_at,
                    tf.where(current_epoch <= lr_small_at,
                        (current_epoch-lr_base_at) *(lr_small-lr_base)/(lr_small_at-lr_base_at)+lr_base,
                        (current_epoch-lr_small_at)*(0-lr_small)/(EPOCHS-lr_small_at)+lr_small,
                    )
            )

        elif S("optimizer.decay.type") == "superconvergence_x":
            lr_small = 0.25*lr_base
            lr_small_at = 0.6*EPOCHS

            s = 1/lr_base_at
            t = lr_base*np.exp(1)/lr_base_at
            learning_rate = t*current_epoch*tf.exp(-current_epoch*s)

        elif S("optimizer.decay.type") == "resnet":
            # base_lr = .128
            base_lr = .256

            learning_rate = learning_rate_with_decay(
                # batch_size=1.0*S("batches.size"), batch_denom=S("batches.size"),
                batch_size=1.0*S("batches.size"), batch_denom=256,
                # num_images=1.0*sampler.num_examples_per_epoch("train"), 
                boundary_epochs=[30, 60, 80, 90],
                # boundary_epochs=[0.3, .60, .80, .90],
                decay_rates=[1, 0.1, 0.01, 0.001, 1e-4], warmup=True, base_lr=base_lr
            )(tf.train.get_or_create_global_step())
        else:
            raise ValueError("optimizer.decay.type '"+S("optimizer.decay.type")+"' not known")
        tf.summary.scalar("learning_rate",learning_rate)

        # optimizer
        spam_spec = importlib.util.find_spec('.'.join([S("model.type"), "optimizer"]))
        if not S("test_only"):
            with tf.name_scope("optimizer"):
                if spam_spec is not None:
                    if S("optimizer.use_custom"):
                        optimizer = importlib.import_module('.'.join([S("model.type"), "optimizer"])).optimizer(learning_rate=learning_rate,epsilon=learning_epsilon)
                    else:
                        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,epsilon=learning_epsilon,beta1=S("optimizer.adam.beta1"),beta2=S("optimizer.adam.beta2"))
                else:
                    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,epsilon=learning_epsilon,beta1=S("optimizer.adam.beta1"),beta2=S("optimizer.adam.beta2"))
                    optimizer = tf.contrib.opt.AdamWOptimizer(learning_rate=learning_rate,epsilon=learning_epsilon,beta1=S("optimizer.adam.beta1"),beta2=S("optimizer.adam.beta2"),weight_decay=S("optimizer.weight_decay"))
                    # optimizer = tf.contrib.opt.MomentumWOptimizer(learning_rate=learning_rate, momentum=S("optimizer.momentum.momentum"), use_nesterov=S("optimizer.momentum.nesterov"), weight_decay=S("optimizer.weight_decay"))

        # network
        scopes_itself = hasattr(networks,"scopes_itself") and networks.scopes_itself
        if not S("test_only"):
            print("initializing train network")
            with tf.variable_scope(tf.get_variable_scope() if scopes_itself else "network") as vs:
                net = network(*data,mode=tf.estimator.ModeKeys.TRAIN)
            if S("validation"):
                print("initializing validation network")
                GLOBAL["weight_counter"] = 0
                with tf.name_scope("network_val") as vs:
                    reset_graph_uids()
                    with tf.variable_scope(tf.get_variable_scope() if scopes_itself else "network", reuse=True) as vs:
                        net_val = network(*validation_data,mode=tf.estimator.ModeKeys.EVAL)

            if S("util.tfl") == "tf_mod" and not S("attention_predict"):
                print("manipulating train-graph")
                fold_batch_norms.FoldBatchNorms(tf.get_default_graph(), is_training=True)
        if not S("train_only"):
            print("initializing test network")
            GLOBAL["weight_counter"] = 0
            with tf.name_scope(tf.get_default_graph().get_name_scope() if S("test_only") else "network_test") as ns:
                reset_graph_uids()
                with tf.variable_scope(tf.get_variable_scope() if scopes_itself else "network", reuse=not S("test_only")) as vs:
                    net_test = network(*data,mode=tf.estimator.ModeKeys.PREDICT)

                if S("util.tfl") == "tf_mod" and not S("attention_predict"):
                    print("manipulating test-graph")
                    fold_batch_norms.FoldBatchNorms(tf.get_default_graph(), is_training=False)


        # loss
        if not S("test_only"):
            with tf.name_scope("loss"):
                loss = lossfn(net, *data, mode=tf.estimator.ModeKeys.TRAIN)
            if S("validation"):
                with tf.name_scope("loss_validation"):
                    loss_val = lossfn(net_val, *validation_data, mode=tf.estimator.ModeKeys.EVAL)
        if not S("train_only"):
            loss_test = lossfn(net_test, *data, mode=tf.estimator.ModeKeys.PREDICT)

        # metrics
        def make_accuracy(net, data):
            with tf.name_scope('accuracy'):
                with tf.name_scope("output"):
                    logits = tf.identity(net, name='logits')
                    labels = tf.identity(data[1], name='labels')

                with tf.name_scope("metrics"):

                    # accuracy
                    with tf.name_scope('correct_prediction'):
                        correct_prediction = tf.equal(tf.argmax(net, 1), tf.cast(labels, tf.int64))
                    correct_prediction = tf.cast(correct_prediction, tf.float32)
                    accuracy = tf.reduce_mean(correct_prediction)
                    tf.summary.scalar("accuracy",accuracy)

                    # compute top_k accuracies
                    for k in range(1, 6, 2):
                        topk = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, k), tf.float32),
                                name='top_' + str(k) + '_accuracy')
                        tf.summary.scalar('top_' + str(k) + '_accuracy', topk)
            return accuracy, correct_prediction

        if not S("test_only"):
            with tf.name_scope("train"):
                accuracy, correct_prediction = make_accuracy(net, data)

                # optimization operation
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())
                do_train_batch = tf.group([train_op, update_ops])
                # TODO how to update batch-norm if optimizer user multiple evaluation steps?!

            if S("validation"):
                with tf.name_scope("validation"):
                    make_accuracy(net_val, validation_data)

        if not S("train_only"):
            with tf.name_scope("test"):
                accuracy_test, correct_prediction_test = make_accuracy(net_test, data)


        # ========== #
        # Additional #
        # ========== #

        # Pruning
        # -------
        if S("pruning.activate"):
            pruning = tf.contrib.model_pruning.Pruning(sparsity=S("pruning.sparsity"))
            prune_op = pruning.mask_update_op()


        # ======== #
        # TRAINING #
        # ======== #
        if not S("test_only"):

            # evaluate these tensors periodically
            estimated_finish = tf.cast(tf.round(STEPS_TOTAL*(1.0*tf.timestamp()-time.time())/(tf.cast(tf.train.get_or_create_global_step(),tf.float64))),tf.int64)
            estimated_finish_formatted = tf.string_join([tf.as_string(estimated_finish // 3600),"h,",tf.as_string((estimated_finish % 3600) // 60),"m"])
            logtensors = {
                "step": tf.train.get_or_create_global_step(),
                "L": loss,
                "acc": accuracy,
                "lr": learning_rate,
                "epoch[%]": tf.cast(tf.round(GLOBAL["global_step"]%STEPS_PER_EPOCH/STEPS_PER_EPOCH*100),tf.int64),
                "epoch[#]": tf.cast(GLOBAL["global_step"]/STEPS_PER_EPOCH,tf.int64),
                "est. end": estimated_finish_formatted,
                "current_epoch": current_epoch,
            }

            # define all hooks
            hks = [
                # hook to initialize data iterators
                # iterator are initialized by placeholders
                # so we need to feed them during init
                IteratorInitializerHook(lambda s: s.run(
                    train_ds_it_init
                ))

            ]

            # hook to save the summaries
            if S("log.summaries"):

                if len(tf.get_collection(tf.GraphKeys.SUMMARIES)) > 0:
                    hks.append(CustomSummarySaverHook(
                        save_steps=int(S("log.summary.steps"))+ 2,
                        # save_steps=1,
                        summary_op=tf.summary.merge_all(),
                        output_dir=S("log.dir")
                    ))
                else:
                    print("No summaries given.")

                # evaluate validation_data
                if S("validation"):
                    if len(tf.get_collection("SUMMARIES_VALIDATION")) > 0:
                        hks.append(CustomSummarySaverHook(
                            save_steps=int(S("log.summary.steps_validation"))+ 2,
                            # save_steps=1,
                            summary_op=tf.summary.merge_all("SUMMARIES_VALIDATION"),
                            output_dir=S("log.dir")
                        ))
                    else:
                        print("No validation-summaries given.")

            # hook to save the model
            hks.append(tf.train.CheckpointSaverHook(
                S("log.dir"),
                save_secs=60 * S("log.checkpoints.eachmins")
            ))

            # hook to get logger output
            hks.append(tf.train.LoggingTensorHook(
                logtensors,
                every_n_iter=int(S("log.console.steps"))
            ))

            hks.append(OnceSummarySaverHook(
                summary_op=tf.summary.merge_all("SUMMARIES_ONCE"),
                output_dir=S("log.dir")
            ))

            smsargs = {"hooks": hks, "config":tf.ConfigProto(log_device_placement=S("log_device_placement"))}

            # restores checkpoint and continues training
            if S("log.checkpoints.restore"):
                smsargs["checkpoint_dir"] = S("log.dir")

            with tf.train.SingularMonitoredSession(**smsargs) as sess:
                print(80 * '#')
                print('#' + 34 * ' ' + ' TRAINING ' + 34 * ' ' + '#')
                print(80 * '#')

                # skip for log (optional)
                if S("optimizer.use_custom") and S("optimizer.memory_size") > 1:
                    summary, logger = hks[0], hks[2]
                    summary._timer._last_triggered_step = S("optimizer.memory_size")/2
                    # summary._timer._last_triggered_step = 1
                    logger._timer._last_triggered_step  = S("optimizer.memory_size") - 1

                if hasattr(optimizer,"loop"):

                    while not sess.should_stop():
                        optimizer.loop(sess, do_train_batch, summary)
                else:
                    while not sess.should_stop():
                        _ = sess.run(do_train_batch)

        if S("train_only"):
            sys.exit(0)



        # ======= #
        # TESTING #
        # ======= #


        # evaluate these tensors periodically
        logtensors = {
            # "accuracy": accuracy_test
            # "net_test": tf.argmax(net_test, 1),
            # "labels": data[1],
        }

        # define all hooks
        hks = [
            # hook to get logger output
            # tf.train.LoggingTensorHook(
            #     logtensors,
            #     every_n_iter=1
            # ),
            # hook to initialize data iterators
            # iterator are initialized by placeholders
            # so we need to feed them during init
            IteratorInitializerHook(lambda s: s.run(
                test_ds_it_init
            )),

            # CustomSummarySaverHook(
            #     save_steps=1,
            #     # save_steps=1,
            #     summary_op=tf.summary.merge_all(),
            #     output_dir=S("log.dir")+"_test"
            #     # output_dir=S("log.dir")
            # )
        ]

        # pruning
        if S("pruning.activate"):
            hks.append(IteratorInitializerHook(lambda s: s.run(
                prune_op
            )))

        # define new scaffold
        if S("log.optimistic_restore"):
            ckpt_file = os.path.join(S("log.dir"),"model.ckpt")
            optimistic_saver = optimistic_restore(ckpt_file)
            # opt_saver.restore(session, save_file)
            scaffold = tf.train.Scaffold(
                init_fn=lambda self,sess: optimistic_saver.restore(sess,ckpt_file),
                saver=optimistic_saver,
            )
        else:
            scaffold = None


        # ------------- #
        # testing loops #
        # ------------- #

        # save image statistics
        if S("preact_stats_first_batch") or S("preact_stats_by_image"):
            loops.save_preact_img_stats(locals())
            sys.exit(0)

        # predict only patches
        if S("predict_patches"):
            loops.get_accuracy_for_batches(locals())
            sys.exit(0)

        # predict twice: low-sample mode for attention, high-sample mode for result
        if S("attention_predict"):
            loops.attention_predict(locals())
            sys.exit(0)

        # default loop
        loops.get_accuracy(locals())

    # catch KeyboardInterrupt error message
    # IT WAS INTENTIONAL
    except KeyboardInterrupt:
        pass
