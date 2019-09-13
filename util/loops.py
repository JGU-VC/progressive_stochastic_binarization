
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pickle
from py.welford import Welford
from template.misc import GLOBAL
import pickle
from template.misc import print_info as print, CustomSummarySaverHook
from tensorflow.contrib.quantize.python import common
from util import fold_batch_norms


def save_preact_img_stats(local):
    # globals().update(local)
    data, S, loss_test, hks, scaffold = local["data"], local["S"], local["loss_test"], local["hks"], local["scaffold"]

    print("getting preactivations tensors")
    preact_tensors = [o.inputs[0] for o in tf.get_default_graph().get_operations() if o.name.endswith("Relu")] # and tf.gradients(o.inputs[0],data[0])[0] is not None]
    # preact_tensors = [preact_tensors[0],preact_tensors[-1]]
    # print(preact_tensors)
    # print("getting gradients of these")
    # preact_tensors_grads = [tf.gradients(loss_test,o) for o in preact_tensors]
    print("initializing tensors")
    preact_mean = [ np.zeros([S("batches.size")]+o.shape.as_list()[1:]) for o in preact_tensors]
    preact_var = [ np.zeros([S("batches.size")]+o.shape.as_list()[1:]) for o in preact_tensors]

    APPROX_ERROR = True
    APPROX_ERROR_ABSOLUTE = False
    if APPROX_ERROR:
        print("loading activations of real network")
        with open("preacts_"+(S("model.classification_models.model") if S("model.type") == "model.classification_models" else S("model.type"))+"_real.bin","rb") as f:
            #img_np, preact_mean, preact_var, preact_grad = pickle.load(f)
            img_np_real, preact_mean_real, preact_var_real = pickle.load(f)
            # preact_mean_real = [preact_mean_real[0],preact_mean_real[-1]]
            # preact_var_real = [preact_var_real[0],preact_var_real[-1]]

    # get last spatial layer + entropy + mask
    print(preact_tensors[-1])
    last_spatial        = preact_tensors[-1]
    pixelwise_ce = tf.losses.softmax_cross_entropy(last_spatial,last_spatial, reduction=tf.losses.Reduction.NONE)
    pixelwise_ce = tf.expand_dims(pixelwise_ce,axis=-1)
    mask = tf.cast(pixelwise_ce > tf.reduce_mean(pixelwise_ce,axis=[1,2],keepdims=True),tf.float32)

    print("ESTIMATING !!!!")

    with tf.train.SingularMonitoredSession(
        scaffold=scaffold,
        hooks=hks,  # list of all hooks
        checkpoint_dir=None if S("log.optimistic_restore") else S("log.dir")  # restores checkpoint
    ) as sess:
        # loop_size = 1
        # loop_size = 100
        # loop_size = 4
        is_real = S("util.variable.transformation") == GLOBAL["transformation_templates"]["real"]
        if is_real:
            APPROX_ERROR = False
        loop_size = 100 if not is_real else 1
        # loop_size = 10 if not is_real else 1
        # loop_size = 1 if not is_real else 1
        image_np = sess.run(data[0])
        entropy_np = sess.run(pixelwise_ce)
        mask_np = sess.run(mask)

        pbar = tqdm(total=loop_size)

        # estimate mean
        preacts_np = [Welford() for i in range(len(preact_tensors))]
        preact_mean, preact_var = [], []
        for i in range(loop_size):
            preacts = sess.run(preact_tensors)
            for j,p in enumerate(preacts):
                if APPROX_ERROR:
                    p_real = preact_mean_real[j]
                    if APPROX_ERROR_ABSOLUTE:
                        preacts_np[j](np.abs((p_real-p)))
                    else:
                        preacts_np[j](np.abs((p_real-p)/(p+1e-7)))
                else:
                    if np.isnan(p).any():
                        print(str(j)+"contains nan")
                    preacts_np[j](p)
            pbar.update(1)
        for j,p in enumerate(preacts_np):
            if np.isnan(p.mean).any():
                print(str(j)+"(mean) contains nan")
            preact_mean.append(p.mean)
            preact_var.append(p.var)

        typename = "real" if is_real else "binom"+str(S("binom.sample_size"))+"_sample"+str(loop_size)
        with open("preacts_"+(S("model.classification_models.model") if S("model.type") == "model.classification_models" else S("model.type"))+"_"+typename+".bin","wb") as f:
            save_tensors = [image_np,preact_mean,preact_var,entropy_np,mask_np] #,preact_mean_grad]
            pickle.dump(save_tensors,f)

    pbar.close()


def get_accuracy_for_batches(local):
    # get needed global variables
    hks, scaffold, test_size, net_test, data, print_orig, S, GLOBAL = local["hks"], local["scaffold"], local["test_size"], local["net_test"], local["data"], local["print_orig"], local["S"], local["GLOBAL"]

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

        return correct_prediction

    num_patches = GLOBAL["patches"]
    data = data[0], tf.split(data[1],num_patches)[0]

    # get network result without softmax
    with tf.name_scope("patches_collect"):
        avg_pool            = net_test.op.inputs[1].op.inputs[0].op.inputs[0].op.inputs[0].op
        last_spatial        = net_test.op.inputs[1].op.inputs[0].op.inputs[0].op.inputs[0].op.inputs[0]
        patches_concat      = tf.concat(tf.split(last_spatial, num_patches),axis=2)
        patches_concat_test = tf.concat(tf.split(data[0], num_patches),axis=2)
        tf.summary.image("patches_concat_in",patches_concat_test)
        tf.summary.image("patches_concat_out",tf.reduce_max(patches_concat,axis=-1,keepdims=True))
        avg_new = tf.reduce_mean(patches_concat, axis=[1,2],name="avg_new")
        # avg_new = tf.reduce_max(patches_concat, axis=[1,2],name="avg_new")
        avg_new = tf.concat([avg_new]*num_patches,axis=0)
        nodes_modified_count = common.RerouteTensor(avg_new, avg_pool.outputs[0])
        if nodes_modified_count == 0:
          raise ValueError('Replacing failed.')

    net_test = tf.split(net_test,num_patches)[0]
    correct_prediction_test = make_accuracy(net_test,data)

    accuracy_res = 0
    # steps = 0
    i = 0

    # for new summaries
    hks.append(CustomSummarySaverHook(
            save_steps=1,
            # save_steps=1,
            summary_op=tf.summary.merge_all(),
            output_dir=S("log.dir")+"_test"
            # output_dir=S("log.dir")
        ))

    with tf.train.SingularMonitoredSession(
            scaffold=scaffold,
            hooks=hks,  # list of all hooks
            checkpoint_dir=None if S("log.optimistic_restore") else S("log.dir")  # restores checkpoint
    ) as sess:
        print(80 * '#')
        print('#' + 34 * ' ' + ' TESTING ' + 35 * ' ' + '#')
        print(80 * '#')
        pbar = tqdm(total=test_size)
        while not sess.should_stop():
            # print(sess.run(data[1]))
            correct = sess.run(correct_prediction_test)
            i += correct.shape[0]
            pbar.update(correct.shape[0])
            accuracy_current = np.sum(correct)
            accuracy_res += accuracy_current

            pbar.set_description("∅-Acc %f, current Acc %f" % ((accuracy_res / i),accuracy_current/correct.shape[0]))
            # pbar.set_postfix("current Acc %f" % accuracy_current)
    print("Total Accuracy:",accuracy_res / i, i)
    pbar.close()

    # for easier grepping using bash-scripts
    print_orig(accuracy_res / i)


def reduce_img(img):
    num_channels = img.shape.as_list()[-1]
    if num_channels != 3 and num_channels != 1:
        img = tf.reduce_sum(img,axis=-1,keepdims=True)
    return img

def attention_predict(local):
    # get needed global variables
    hks, scaffold, test_size, print_orig, net_test, data, S, make_accuracy = local["hks"], local["scaffold"], local["test_size"], local["print_orig"], local["net_test"], local["data"], local["S"], local["make_accuracy"]

    # convert last spatial layer to mask

    # resnet50_v2
    # last_spatial        = net_test.op.inputs[1].op.inputs[0].op.inputs[0].op.inputs[0].op.inputs[0]

    # resnet18_slim
    last_spatial          = net_test.op.inputs[0].op.inputs[0].op.inputs[0].op.inputs[0].op.inputs[0].op.inputs[0]
    print(last_spatial)
    # fl_weight           = net_test.op.inputs[1].op.inputs[0].op.inputs[0].op.inputs[1]
    # fl_bias             = net_test.op.inputs[1].op.inputs[0].op.inputs[1]
    # with tf.variable_scope("attention_psb"):
    #     last_spatial        = tf.nn.conv2d(last_spatial,tf.reshape(fl_weight,[1,1]+fl_weight.shape.as_list()),strides=[1]*4,padding="SAME", name="additional_psb") + fl_bias
    fraction = S("attention.fraction")
    img_shape = data[0].shape.as_list()[1:3]
    mask_shape = last_spatial.shape.as_list()[1:3]

    if S("attention.mode") != "neuron":

        if S("attention.spatial_mode") == "random":
            mask_np = 1.0*(np.random.random([1]+mask_shape+[1]) < fraction)
            mask = tf.constant(mask_np,tf.float32)

        elif S("attention.spatial_mode") == "center":
            mask_np = np.zeros([1]+mask_shape+[1])
            mask_np[0,3,3,0] = 1
            mask = tf.constant(mask_np,tf.float32)

        if S("attention.spatial_mode") == "max_activation":
            activation_per_pixel      = tf.reduce_max(last_spatial,axis=-1,keepdims=True)
            image_max          = tf.reduce_max(last_spatial,axis=[1,2,3],keepdims=True)
            mask = tf.cast(tf.equal(activation_per_pixel, image_max),tf.float32)

        elif S("attention.spatial_mode") == "mean_activation":
            activation_per_pixel      = tf.reduce_mean(last_spatial,axis=-1,keepdims=True)
            image_mean          = tf.reduce_mean(last_spatial,axis=[1,2,3],keepdims=True)
            mask = tf.cast(activation_per_pixel > image_mean*fraction,tf.float32)

        elif S("attention.spatial_mode") == "mean_entropy":
            pixelwise_ce = tf.losses.softmax_cross_entropy(last_spatial,last_spatial, reduction=tf.losses.Reduction.NONE)
            pixelwise_ce = tf.expand_dims(pixelwise_ce,axis=-1)
            mask = tf.cast(pixelwise_ce > tf.reduce_mean(pixelwise_ce,axis=[1,2],keepdims=True)*fraction,tf.float32)

        elif S("attention.spatial_mode") == "max_entropy":
            pixelwise_ce = tf.losses.softmax_cross_entropy(last_spatial,last_spatial, reduction=tf.losses.Reduction.NONE)
            activation_per_pixel      = pixelwise_ce
            image_max                 = tf.reduce_max(pixelwise_ce,axis=[1,2],keepdims=True)
            pixelwise_ce = tf.expand_dims(pixelwise_ce,axis=-1)
            image_max = tf.expand_dims(image_max,axis=-1)
            mask = tf.cast(tf.equal(pixelwise_ce, image_max),tf.float32)


        if S("attention.spatial_surround") > 1:
            mask = tf.layers.max_pooling2d(mask,pool_size=S("attention.spatial_surround"),padding="same",strides=1)

        # top k patches
        # # k = 8
        # k = 15
        # # pixelwise_ce = tf.layers.average_pooling2d(pixelwise_ce,pool_size=3,padding="valid",strides=1)
        # tf.summary.image("mask",reduce_img(data[0]*mask_scaled+data[0]*(1-mask_scaled)*0.5))
        # tf.summary.image("entropy",reduce_img(pixelwise_ce))
        # ce_shape = pixelwise_ce.shape.as_list()[1:3]
        # pixelwise_ce = tf.layers.flatten(pixelwise_ce)
        # top_k_val, top_k_ind = tf.nn.top_k(pixelwise_ce,k)
        # mask = tf.reduce_sum([
        #     tf.one_hot(top_k_ind[:,i],depth=pixelwise_ce.shape.as_list()[-1])
        #     for i in range(k)
        # ], axis=0)
        # mask = tf.reshape(mask,[-1]+ce_shape+[1])

        # plot mask
        mask_scaled = tf.image.resize_images(mask, img_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        tf.summary.image("mask",reduce_img(data[0]*mask_scaled+data[0]*(1-mask_scaled)*0.3))

    # initialize mask-counter
    if S("attention.mode") == "spatial" or S("attention.mode") == "spatial_old":
        mask_sum = tf.reduce_sum(mask)
        mask_total = tf.reduce_sum(mask*0+1)
    elif S("attention.mode") == "channels":
        mask_sum = 0
        mask_total = 0

    # fold batch norms, replace weights, ...
    if S("util.tfl") == "tf_mod":
        print("manipulating original graph")
        fold_batch_norms.FoldBatchNorms(tf.get_default_graph(), is_training=False)


    if S("attention.mode") == "neuron":
        mask_sum = GLOBAL["m_sum"]
        mask_total = GLOBAL["m_total"]
        accuracy_test_masked, correct_prediction_test = make_accuracy(net_test, data)

    else:
        # reuse model (tf_resnet_official)
        with tf.variable_scope(tf.get_variable_scope(),reuse=True): # for tf_resnet_official
            logits_masked = GLOBAL["keras_model"](GLOBAL["keras_model_preprocess"](data[0]))
        net_test_masked = logits_masked

        # reuse model (keras)
        # logits_masked = GLOBAL["keras_model"](GLOBAL["keras_model_preprocess"](data[0]))
        # net_test_masked = tf.concat([tf.expand_dims(logits_masked[:,0]*0,1),logits_masked],axis=-1)

        accuracy_test_masked, correct_prediction_test = make_accuracy(net_test_masked, data)

        # new settings
        transformation_template = S("attention.transform")
        if transformation_template == "psb":
            S("binom.sample_size",set=S("attention.sample_size"))
        S("util.variable.transformation",set=GLOBAL["transformation_templates"][transformation_template])
        S("util.variable.transformation.template_name",set=transformation_template)

        # fold batch norms, replace weights, ...
        if S("util.tfl") == "tf_mod":
            print("manipulating attention graph")
            fold_batch_norms.FoldBatchNorms(tf.get_default_graph(), is_training=False)

        print("decide which graph to use per layer")
        from util.fold_batch_norms import _FindRestFilters, _CloneWithNewOperands
        graph = tf.get_default_graph()
        matches = _FindRestFilters(graph,False)
        print("Replacing",len(matches),"Conv|Mul|DepthwiseConv2dNative-Filters (without a suceeding BatchNorm)")
        for match in matches:
            scope, sep, _ = match['layer_op'].name.rpartition('/')
            model_name = S("model.classification_models.model")+"/"
            if not scope.startswith(model_name):
                continue
            with graph.as_default(), graph.name_scope(scope + sep):
                with graph.name_scope(scope + sep + '_masked' + sep):
                    weight = match['weight_tensor']
                    input_tensor  = match['input_tensor']
                    if not len(input_tensor.shape.as_list())==4:
                        continue
                    kernel_size = weight.shape.as_list()[0]

                    if not input_tensor.name.startswith(model_name):
                        input_tensor_orig = input_tensor
                    else:
                        input_tensor_orig = graph.get_tensor_by_name(input_tensor.name[len(model_name):])
                    output_tensor = match['output_tensor']
                    output_tensor_orig = graph.get_tensor_by_name(output_tensor.name[len(model_name):])

                    img_shape_in = input_tensor.shape.as_list()[1:3]
                    img_shape_out = output_tensor.shape.as_list()[1:3]

                    # add mask to input (and redefine borders)
                    if S("attention.mode") == "spatial_old":
                        mask_scaled2 = tf.image.resize_images(mask, img_shape_in, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                        new_input_tensor = input_tensor*mask_scaled2 + input_tensor_orig*(1-mask_scaled2)
                        new_layer_tensor = _CloneWithNewOperands( match['layer_op'], new_input_tensor, weight, False)
                    elif S("attention.mode") == "spatial":
                        mask_scaled2 = tf.image.resize_images(mask, img_shape_out, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                        output_tensor_new = _CloneWithNewOperands( match['layer_op'], input_tensor , weight, False) # just for rerouting
                        new_layer_tensor = output_tensor_new*mask_scaled2 + output_tensor_orig*(1-mask_scaled2)
                    elif S("attention.mode") == "channels":
                        if not weight.name.startswith(model_name):
                            weight_p = GLOBAL["weights_p"][("/".join(weight.name.split("/")[0:-1])+"/var/p_1:0").replace("kernel","_psb")]
                        else:
                            weight_p = GLOBAL["weights_p"]["/".join(weight.name.split("/")[1:-1])+"/var/p_1:0"]
                        weight_p_mean = tf.reduce_mean(weight_p,axis=[0,1,2],keepdims=True)
                        weight_p_mean_total = tf.reduce_mean(weight_p,keepdims=True)
                        mask_channels =  tf.cast(weight_p_mean > weight_p_mean_total,tf.float32)
                        # mask_channels = tf.transpose(mask_channels,[2,0,1,3])
                        output_tensor_new = _CloneWithNewOperands( match['layer_op'], input_tensor , weight, False) # just for rerouting
                        new_layer_tensor = output_tensor_new*mask_channels + output_tensor_orig*(1-mask_channels)
                        mask_sum += tf.reduce_sum(mask_channels)
                        mask_total += tf.reduce_sum(0*mask_channels+1)

                    # reroute tensor to output depending on sampling mode
                    nodes_modified_count = common.RerouteTensor(new_layer_tensor,output_tensor)

                    if kernel_size > 1:
                        pass
                        # tf.summary.image("mask",reduce_img(input_tensor*mask_scaled2))
                        # tf.summary.image("img_masked",reduce_img(new_input_tensor))

                        # tf.summary.image("input_tensor_all",[
                        #     # tf.reduce_max((new_input_tensor[0]-input_tensor_orig[0])*mask_scaled2[0],axis=-1,keepdims=True),
                        #     # tf.reduce_max(input_tensor[0],axis=-1,keepdims=True),
                        #     tf.reduce_max(tf.abs(input_tensor[0]-input_tensor_orig[0]),axis=-1,keepdims=True),
                        #     tf.reduce_max(mask_scaled2[0],axis=-1,keepdims=True),
                        #     tf.reduce_max(input_tensor_orig[0],axis=-1,keepdims=True),
                        #     tf.reduce_max(input_tensor[0],axis=-1,keepdims=True),
                        #     tf.reduce_max(new_input_tensor[0],axis=-1,keepdims=True)
                        # ], max_outputs=4)

                    if nodes_modified_count == 0:
                        raise ValueError('Folding batch norms failed, %s had no outputs.' % match['output_tensor'].name)

    # for new summaries
    hks.append(CustomSummarySaverHook(
            save_steps=1,
            # save_steps=1,
            summary_op=tf.summary.merge_all(),
            output_dir=S("log.dir")+"_test"
            # output_dir=S("log.dir")
        ))

    correct_prediction_test_mask = correct_prediction_test
    correct_prediction_test = local["correct_prediction_test"]

    accuracy_res = 0
    mask_sum_np, mask_total_np = 0, 0
    # steps = 0
    i = 0
    with tf.train.SingularMonitoredSession(
            scaffold=scaffold,
            hooks=hks,  # list of all hooks
            checkpoint_dir=None if S("log.optimistic_restore") else S("log.dir")  # restores checkpoint
    ) as sess:
        print(80 * '#')
        print('#' + 34 * ' ' + ' TESTING ' + 35 * ' ' + '#')
        print(80 * '#')
        pbar = tqdm(total=test_size)
        while not sess.should_stop():
            print("run",i)
            correct,mask_sum_np_c,mask_total_np_c = sess.run([correct_prediction_test_mask,mask_sum,mask_total])
            mask_sum_np += mask_sum_np_c
            mask_total_np += mask_total_np_c
            i += correct.shape[0]
            pbar.update(correct.shape[0])
            accuracy_current = np.sum(correct)
            accuracy_res += accuracy_current

            pbar.set_description("∅-Acc %f, current Acc %f, mask-proportion %f" % ((accuracy_res / i),accuracy_current/correct.shape[0], mask_sum_np/mask_total_np if mask_total_np > 0 else "nothing masked"))
            # pbar.set_postfix("current Acc %f" % accuracy_current)
    # print("Total Accuracy:",accuracy_res / i, i)
    print("Total Proportion:",mask_sum_np/mask_total_np, mask_sum_np, mask_total_np)
    pbar.close()

    # for easier grepping using bash-scripts
    print_orig(mask_sum_np / mask_total_np)
    print_orig(accuracy_res / i)




def get_accuracy(local):
    # get needed global variables
    hks, scaffold, test_size, correct_prediction_test, print_orig, S = local["hks"], local["scaffold"], local["test_size"], local["correct_prediction_test"], local["print_orig"], local["S"]

    accuracy_res = 0
    # steps = 0
    i = 0
    with tf.train.SingularMonitoredSession(
            scaffold=scaffold,
            hooks=hks,  # list of all hooks
            checkpoint_dir=None if S("log.optimistic_restore") else S("log.dir")  # restores checkpoint
    ) as sess:
        print(80 * '#')
        print('#' + 34 * ' ' + ' TESTING ' + 35 * ' ' + '#')
        print(80 * '#')
        pbar = tqdm(total=test_size)
        while not sess.should_stop():
            correct = sess.run(correct_prediction_test)
            i += correct.shape[0]
            pbar.update(correct.shape[0])
            accuracy_current = np.sum(correct)
            accuracy_res += accuracy_current

            pbar.set_description("∅-Acc %f, current Acc %f" % ((accuracy_res / i),accuracy_current/correct.shape[0]))
            # pbar.set_postfix("current Acc %f" % accuracy_current)
    print("Total Accuracy:",accuracy_res / i, i)
    pbar.close()

    # for easier grepping using bash-scripts
    print_orig(accuracy_res / i)


