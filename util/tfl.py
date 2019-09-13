from template.misc import S
S = S(scope="util")

if S("tfl") == "tf" or S("tfl") == "tf_mod":
    import tensorflow.layers as tfl
elif S("tfl") == "custom":
    import util.tfl_custom as tfl
else:
    raise ValueError("util.tfl-library '"+S("tfl")+"'")


