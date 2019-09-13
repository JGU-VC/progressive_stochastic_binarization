import tensorflow as tf
from settings_validator import validate as validate_settings
from settings import SETTINGS
import numpy as np
from colorsys import hls_to_rgb





# ======== #
# settings #
# ======== #

# useage:
#   get a variable directly
#       val = S("group.subgroup.var")
#   alternatively, define the subgroup first
#       S = S(scope="group.subgroup")
#       val = S("var")
#
#   get variable, with default
#       val = S("non_existant_variable",alt="alternative value")
#

def getS(key,scope="",alt=None,set=None):
    if len(scope) > 0:
        key = scope+"."+key
    keys = key.split(".")
    v = SETTINGS

    if set is not None:
        for k in keys[0:-1]:
            try:
                v = v[k]
            except:
                if alt is not None:
                    return alt
                else:
                    raise KeyError("'"+k+"' not found in '"+key+"'")
        v[keys[-1]] = set
        return


    for k in keys:
        try:
            v = v[k]
        except:
            if alt is not None:
                return alt
            else:
                raise KeyError("'"+k+"' not found in '"+key+"'")
    return v


# SETTINGS-getter
def S(key=None,scope="",alt=None,set=None):
    if key is not None:
        return getS(key,scope,alt,set)
    else:
        def newS(key=None,scope=scope,alt=alt,set=None):
            return S(key,scope,alt,set)
        return newS




# ================ #
# Global Variables #
# ================ #

# eval_step = tf.get_variable("eval_step",initializer=-tf.ones(()), trainable=False)
# memory_step = tf.cast(eval_step % S("optimizer.memory_size"),tf.int64, name="memory_step")
# eval_step_increase = tf.assign_add(eval_step, 1, name="eval_step_increase")
GLOBAL = {
    # "eval_step": eval_step,
    # "eval_step_increase": eval_step_increase,
    # "memory_step": memory_step,
    "global_step": tf.cast(tf.train.get_or_create_global_step(),tf.float32),

    "weight_counter": 0,
    "weights": [],
    "weights_p": {},
    "reuse_mode": False, # for summaries
    "preactivations": []
}




# ========= #
# arguments #
# ========= #

# add settings-object to argparse
def settings_add_to_argparse(SObj,parser,prefix=""):
    if len(prefix) > 0:
        prefix = prefix+"."
    for k,v in SObj.items():
        if isinstance(v,dict):
            settings_add_to_argparse(v,parser,prefix=prefix+k)
        else:
            varname = prefix+k
            val = S(varname)
            if isinstance(v,bool):
                parser.add_argument('--'+varname, dest=varname, action='store_true', help=S("_"+varname,alt=""))
                parser.add_argument('--no-'+varname, dest=varname, action='store_false')
                parser.set_defaults(**{varname:S(varname)})
            elif isinstance(v,int):
                parser.add_argument( "--"+varname, type=int, default=val, help=S("_"+varname,alt=""))
            elif isinstance(v,str):
                parser.add_argument( "--"+varname, type=str, default=val, help=S("_"+varname,alt=""))
            elif isinstance(v,float):
                parser.add_argument( "--"+varname, type=float, default=val, help=S("_"+varname,alt=""))

# produces markdown & cli-representation of any nested dict
def dict_to_string(SObj, settings_str="", settings_str_tb="", prefix=""):
    max_key_len = 0
    if len(prefix) > 0:
        prefix = prefix+"."
    for i,(k,v) in enumerate(SObj.items()):
        if i==0:
            str_k = prefix+str(k)
        else:
            str_k = " "*(len(prefix)-1)+"."+str(k) if len(prefix)>0 else str(k)
        if isinstance(v,dict):
            heading_k = prefix+str(k)
            heading_sym = "-" if len(prefix)>0 else "="
            heading = "\n" + heading_k.upper()+"\n" + heading_sym*len(heading_k)+"\n"
            # settings_str += heading
            # settings_str += str_k+"\n"
            settings_str_tb += heading

            settings_str, settings_str_tb = dict_to_string(v,settings_str, settings_str_tb, prefix=prefix+k)
        elif isinstance(v,list) and k in ["hidden","weight"]:
            settings_str += str_k+"\n"
            settings_str_tb += "`"+prefix+str(k)+"`"+"\n\n"
            # settings_str_tb += "\n```\n"
            for el in v:
                if isinstance(el,tuple):
                    el_str = str(el[0])+" = "+str(el[1])
                else:
                    varname = "p" if k == "hidden" else "w"
                    el_str = varname + " = "+str(el)
                settings_str += len(str_k)*" "+el_str+"\n"
                settings_str_tb += "- `"+el_str+"`\n"
            settings_str_tb += "\n\n"
        else:
            varname = prefix+k
            settings_str += str_k+" "*(max_key_len-len(k))+" = "+str(S(varname))+"  \n"
            settings_str_tb += "`"+str(varname)+" "*(max_key_len-len(k))+" = "+str(S(varname))+"`  \n"
    return settings_str, settings_str_tb

# overwrites settings by argparse arguments & print them
def print_and_override_settings(SETTINGS,args):
    print("===============================================================")
    print("=                          Settings                           =")
    print("===============================================================")
    max_key_len = 0
    for k,v in vars(args).items():
        if S(k) != v:
            S(k,set=v)
            print(k+" -> overloaded by cli")
        max_key_len = max(max_key_len,len(k))
    validate_settings(S,vars(args).keys(),GLOBAL)
    settings_str, settings_str_tb = dict_to_string(SETTINGS)
    print(settings_str)
    tf.add_to_collection("SUMMARIES_ONCE", tf.summary.text("settings", tf.constant(settings_str_tb), collections="SUMMARIES_ONCE"))
    print("===============================================================")
    print("=                        Settings End                         =")
    print("===============================================================")





# ============= #
# session hooks #
# ============= #

# Define data loaders
# See https://gist.github.com/peterroelants/9956ec93a07ca4e9ba5bc415b014bcca
class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self, func=None):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = func

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)



# redefine summarysaverhook (for more accurate saving)
from tensorflow.python.training.session_run_hook import SessionRunArgs
class CustomSummarySaverHook(tf.train.SummarySaverHook):
    """Saves summaries every N steps."""

    def begin(self):
        super().begin()
        self._timer.reset()
        self._iter_count = 0

    def before_run(self, run_context):	# pylint: disable=unused-argument
        if S("optimizer.use_custom") and S("optimizer.memory_size") > 1:
            self._request_summary = (self._iter_count % S("optimizer.memory_size") == S("optimizer.memory_size") - 1) # and self._timer.should_trigger_for_step(self._next_step)
        else:
            self._request_summary = ( self._next_step is None or
                        self._timer.should_trigger_for_step(self._next_step))
        requests = {"global_step": self._global_step_tensor}
        if self._request_summary:
            if self._get_summary_op() is not None:
                # print(self._iter_count)
                requests["summary"] = self._get_summary_op()

        return SessionRunArgs(requests)

    def after_run(self, run_context, run_values):
        super().after_run(run_context,run_values)
        self._iter_count += 1

class OnceSummarySaverHook(tf.train.SummarySaverHook):
    """Saves summaries every N steps."""

    def __init__(self, output_dir=None, summary_writer=None, scaffold=None, summary_op=None):
        self._summary_op = summary_op
        self._summary_writer = summary_writer
        self._output_dir = output_dir
        self._scaffold = scaffold
        class emptytimer():
            def update_last_triggered_step(*args,**kwargs):
                pass
        self._timer = emptytimer()

    def begin(self):
        super().begin()
        self._done = False

    def before_run(self, run_context):	# pylint: disable=unused-argument
        self._request_summary = not self._done
        requests = {"global_step": self._global_step_tensor}
        if self._request_summary:
            if self._get_summary_op() is not None:
                # print(self._iter_count)
                requests["summary"] = self._get_summary_op()

        return SessionRunArgs(requests)

    def after_run(self, run_context, run_values):
        super().after_run(run_context,run_values)
        self._done = True




def get_distinct_colors(n):

    colors = []

    for i in np.arange(0., 360., 360. / n):
        h = i / 360.
        l = (50 + np.random.rand() * 10) / 100.
        s = (90 + np.random.rand() * 10) / 100.
        colors.append(hls_to_rgb(h, l, s))

    return colors


# ================== #
# Tensorflow Helpers #
# ================== #
from tensorflow.python.keras import backend
from collections import defaultdict
import weakref
def reset_graph_uids():
    graph = tf.get_default_graph()
    # backend.PER_GRAPH_LAYER_NAME_UIDS.setdefault(graph,defaultdict(int))
    backend.PER_GRAPH_LAYER_NAME_UIDS[graph] = defaultdict(int)


# ===== #
# Other #
# ===== #
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
def print_info(*args,color=bcolors.OKGREEN):
    print(color+bcolors.BOLD+"INFO"+bcolors.ENDC+color+" (tf.boilerplate):"+bcolors.ENDC,*args)

