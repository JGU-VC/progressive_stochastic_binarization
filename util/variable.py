import tensorflow as tf
import numpy as np
from template.misc import S, GLOBAL
from util.helpers import pass_gradient, sample_binomial, next_base2, fixed_point
from util.initializer import TransformInitializer


# global variables for mini-parser
G = {
    "tf": tf,
    "np": np,
    "S": S,
    "pass_gradient": pass_gradient,
    "sample_binomial": sample_binomial,
    "next_base2": next_base2,
    "fixed_point": fixed_point,
    "GLOBAL": GLOBAL,
}

S = S(scope="util.variable")

def variableFromSettings(shape,S=S,hiddenVar=None):
    # local variables for mini-parser
    V = {}

    # initializer
    base_initializer = getattr(tf.initializers,S("initializer"))
    initializer = TransformInitializer(base_initializer,S("transformation.init",alt=[]),dtype=getattr(tf,S("dtype")),seed=S("seed"))


    with tf.name_scope("var"):

        var_name = S("name")

        # define variable
        if hiddenVar is None:
            p = tf.get_variable(name=var_name,shape=shape, initializer=initializer, regularizer=None, trainable=True)
        else:
            p = hiddenVar

        # apply pruning
        if S("pruning.activate",scope=""):
            p = tf.contrib.model_pruning.apply_mask(p,scope=tf.contrib.framework.get_name_scope())

        V["p"] = p

        # check for shared tensors
        localvars_used = []
        for T in [S("transformation.hidden",alt=[]), S("transformation.weight",alt=[]), S("transformation.regularizer.weight_transformation",alt=[])]:

            for i,t in enumerate(T):
                localvars = []
                t_orig = t
                if isinstance(t,tuple):
                    if t[0] not in ["w","p","x"]:
                        localvars_used.append(t[0])
                    t = t[1]

                def parse_func(t,fn,string_fn):
                    if not isinstance(string_fn,str):
                        return str(t)
                    if t.startswith(fn+"(") and t.endswith(")"):
                        names = t[len(fn)+1:-1].split(",")
                        t = string_fn(*names)
                    return t

                # parse predefined functions
                t = parse_func(t,"relaxed_binarize_wolog(0±ε)->[0,1]", lambda var:
                    "tf.sigmoid((%s+eps + tf.log(rng) - tf.log(1-rng))/%s)" % (var, "relaxation_temp")
                )
                t = parse_func(t,"relaxed_binarize_wlog(1±ε)->[0,1]", lambda var:
                    "tf.sigmoid((tf.log(tf.abs(%s+eps)) + tf.log(rng) - tf.log(1-rng))/%s)" % (var, "relaxation_temp")
                )
                t = parse_func(t,"gumbel_binarize_wolog(1±ε)->[0,1]", lambda var:
                    "tf.abs(%s) + tf.log(rng) - tf.log(1-rng)" % var
                )
                t = parse_func(t,"where_binarize[0,1]->{0,1}", lambda var:
                    "tf.where(rng <= "+var+", ones, zeros, name='sampled_filter')"
                )
                t = parse_func(t,"pass_through_binarize[0,1]->{-1,1}", lambda var:
                    "pass_gradient("+var+", lambda p, localvars: tf.where(rng <= p, ones, -ones, name='sampled_filter'))"
                )
                t = parse_func(t,"pass_through_binarize[0,1]->{0,1}", lambda var:
                    "pass_gradient("+var+", lambda p, localvars: tf.where(rng <= p, ones, zeros, name='sampled_filter'))"
                )
                t = parse_func(t,"softround", lambda var:
                    var+" - tf.sin(2*np.pi*"+var+")/(2*np.pi)"
                )
                t = parse_func(t,"passed_round", lambda var:
                    "2**pass_gradient("+var+", lambda x: x - tf.sin(2*np.pi*x)/(2*np.pi))"
                )
                t = parse_func(t,"lecun_normalize", lambda var:
                    "tf.identity(("+var+"-tf.nn.moments("+var+",axes=None)[0])/tf.nn.moments("+var+",axes=None)[1]*np.sqrt(1/np.prod("+var+".get_shape().as_list()[:-1])),name=\"lecun\")"
                )
                t = parse_func(t,"lecun_normalize_no_mean", lambda var:
                    "tf.identity(("+var+")/tf.nn.moments("+var+",axes=None)[1]*np.sqrt(1/np.prod("+var+".get_shape().as_list()[:-1])),name=\"lecun\")"
                )

                # get variables
                V["eps"] = 1e-5
                if "ones" in t and "ones" not in V:
                    localvars.append("ones")
                    V["ones"] = tf.ones(shape)
                if "zeros" in t and "zeros" not in V:
                    localvars.append("zeros")
                    V["zeros"] = tf.zeros(shape)
                if "rng" in t and "rng" not in V:
                    localvars.append("rng")
                    V["rng"] = tf.random_uniform(shape, name="rng") # independent

                for var in localvars_used:
                    if var in t and var not in localvars:
                        localvars.append(var)

                # replace localvars
                if "localvars" in t:
                    t = t.replace("localvars",",".join([v+"="+v for v in localvars]))

                # save modified t again
                if isinstance(t_orig,tuple):
                    T[i] = (t_orig[0],t)
                else:
                    T[i] = t

        # hidden variable transformations
        for t in S("transformation.hidden",alt=[]):
            if isinstance(t,tuple):
                name = t[0]
                V[name] = eval(t[1],{**G,**V})
                if name.lower() == "assert":
                    try:
                        assert V[name]
                    except AssertionError:
                        raise AssertionError(t[1])
            else:
                V["p"] = eval(t,{**G,**V})

        # map hidden weight to weight
        V["w"] = p
        for t in S("transformation.weight",alt=[]):
            if isinstance(t,tuple):
                name = t[0]
                V[name] = eval(t[1],{**G,**V})
                if name.lower() == "assert":
                    try:
                        assert V[name]
                    except AssertionError:
                        raise AssertionError(t[1])
            else:
                V["w"] = eval(t,{**G,**V})


    # add regularizer
    if S("regularizer.type") is not None:
        if all(var_name not in s for s in S("regularizer.exclude_names",alt=[])):
            tf.contrib.layers.apply_regularization(getattr(tf.contrib.layers,S("regularizer.type"))(S("regularizer.weight")), [eval(S("regularizer.weight_transformation"),{**G,**V})])
        else:
            print("excluding:",var_name)

    GLOBAL["weight_counter"] += 1

    # return sampled weight / hidden variable - combo
    return V["w"], V["p"]



