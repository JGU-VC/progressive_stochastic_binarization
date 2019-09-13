from settings import SETTINGS
import uuid

def validate(S, varnames, GLOBAL):
    for name in varnames:
        val = S(name)
        if isinstance(val,str) and val.startswith("as "):
            val = val[3:]
            newval = eval(val)
            S(name,set=newval)

    if S("activation.type") != "timerelu" and S("activation.type") != "trsim":
        if "timerelu" in SETTINGS:
            del SETTINGS["timerelu"]

    dir = S("log.dir")
    dirthis = dir.endswith("!")
    print(dir, dirthis)
    if dirthis:
        dir = dir[:-1]
    if not dirthis and not S("debug"):
        S("log.dir",set=dir+"_"+str(uuid.uuid4()).split("-")[0])
    else:
        S("log.dir",set=dir)

    if isinstance(S("util.variable.transformation"),str):
        transformation_template = S("util.variable.transformation")
        S("util.variable.transformation",set=S("util.variable.transformation_templates."+transformation_template))
        S("util.variable.transformation.template_name",set=transformation_template)

    if "transformation_templates" in SETTINGS["util"]["variable"]:
        GLOBAL["transformation_templates"] = SETTINGS["util"]["variable"]["transformation_templates"]
        del SETTINGS["util"]["variable"]["transformation_templates"]

    if S("pruning.activate") and S("util.tfl") == "tfl":
        raise ValueError("Pruning is only possible for setting: util.variable.tfl == 'tfl_custom'")

    if S("predict_patches"):
        if S("batches.patches_size") == 0:
            raise ValueError("For patch-prediction you need to specify patches. (batches.patches_size> 0)")
        if S("batches.test_like_train") == 0:
            raise ValueError("For patch-prediction you need use batches.test_like_train-mode")
