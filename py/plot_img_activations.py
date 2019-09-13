import pickle
import numpy as np
import sys
import os
import seaborn as sns
import matplotlib.pylab as plt


os.makedirs("img_plots",exist_ok=True)

print("loading file")
with open(sys.argv[-1],"rb") as f:
    #img_np, preact_mean, preact_var, preact_grad = pickle.load(f)
    # img_np, preact_mean, preact_var = pickle.load(f)
    img_np, preact_mean, preact_var, entropy, mask = pickle.load(f)
print("done.")

from PIL import Image
import sys
def save_img(A,path,normalize=False):
    im = Image.fromarray(A.astype(np.uint8))
    im = im.resize((224,224))
    im.save(path)

from matplotlib import cm
# colormap = cm.gist_earth
colormap_name = 'coolwarm'
colormap = cm.get_cmap(colormap_name)
eps = 1e-7
fns = {
    "mean": lambda j: preact_mean[j],
    # "var": lambda j: preact_var[j],
    # "ratio": lambda j: preact_var[j]/np.expand_dims(np.max(np.abs(preact_mean[j]),axis=-1),-1)
    # "ratio": lambda j: np.sqrt(preact_var[j])/np.mean(np.abs(preact_mean[j]),axis=-1, keepdims=True)
    # "relerror": lambda j: np.sqrt(np.max(preact_var[j],axis=-1,keepdims=True))/(np.max(preact_mean[j],axis=-1,keepdims=True)+eps)
    # "relerror": lambda j: np.mean(np.sqrt(preact_var[j])/(np.abs(preact_mean[j])+eps),axis=-1,keepdims=True)
    # "relerror": lambda j: np.sqrt(preact_var[j])/(np.abs(preact_mean[j])+eps)
}

HEATMAP=False

LOCAL_NORMALIZE=True
LOCAL_NORMALIZE_LAYERS=[0,27,35,48]
# LOCAL_NORMALIZE_LAYERS=[0]

for name,fn in fns.items():

    print("Plotting "+name)
    vals = [fn(j) for j in range(len(preact_mean))]

    # normalize with global maximum value
    # max_val = 0
    # for p in vals:
    #     max_val = max(np.max(p),max_val)
    # print(name+"_max",max_val)

    global_max = 0.0
    global_min = np.Infinity
    if not LOCAL_NORMALIZE:
        for j,p in enumerate(vals):
            if j in LOCAL_NORMALIZE_LAYERS:
                global_max = np.maximum(np.max(p),global_max)
                global_min = np.minimum(np.min(p),global_min)

    for j,p in enumerate(vals):
        # continue
        # # p /= max_val
        if j != 48 and j != 0:
            continue
        # continue
        if len(LOCAL_NORMALIZE_LAYERS)!=0 and j not in LOCAL_NORMALIZE_LAYERS:
            continue

        # print(p.shape[0])
        if LOCAL_NORMALIZE:
            print("min:",np.sqrt(np.min(p)), "max:", np.sqrt(np.max(p)), "ratio",np.sqrt(np.max(p))/np.sqrt(np.min(p)+eps),"mean_std",np.mean(np.sqrt(p)))
        else:
            print("global min:",np.sqrt(global_min), "max:", global_max, "ratio",np.sqrt(global_max)/np.sqrt(global_max+eps),"mean_std",np.mean(np.sqrt(p)))
        for k in range(p.shape[0]):
            # for d in range(p.shape[-1]):
            #     img = p[k,...,d]
            #     # img = np.mean(p[k],axis=-1)

            #     # normalize
            #     img = img/np.max(img)

            #     # colored
            #     save_img(colormap(img*255), "img_plots/i%i_l%i_d%i_%s.png" % (k,j,d,name))

            img = np.max(p[k],axis=-1)
            # img = np.mean(np.abs(p[k]),axis=-1)
            # if k == 0:
            #     print("min:",np.sqrt(np.min(img)), "max:", np.sqrt(np.max(img)), "ratio",np.sqrt(np.max(img))/np.sqrt(np.min(img)),"mean_std",np.mean(np.sqrt(img)))
            if np.min(img) == np.max(img):
                print("Min = Max for k=",k," j=",j)
                continue
            # img = np.log(img+eps)
            if LOCAL_NORMALIZE:
                img = (img-np.min(img))/(np.max(img)-np.min(img)+eps)
            else:
                if j in LOCAL_NORMALIZE_LAYERS:
                    img = (img-global_min)/(global_max-global_min)
                    # img = np.log(img+eps)
                    # img = (img-np.min(img))/(np.max(img)-np.min(img)+eps)
            # img = (img-np.min(img))/(0.5*np.max(img)-np.min(img))

            # img = img/np.max(img)
            # img = img/max_val
            if HEATMAP:
                # uniform_data = np.random.rand(10, 12)
                plt.tick_params( which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
                ax = sns.heatmap(img, linewidth=0, cmap=colormap_name, rasterized=True)
                plt.show()
                # plt.savefig("activation_%i.eps"%(j),bbox_inches='tight', transparent="True", pad_inches=0)
            else:
                try:
                    save_img(colormap(img)*255, "img_plots/i%i_l%i_mean_%s.png" % (k,j,name))
                except:
                    pass

        # greyscale
        # vals[j] *= 255
        # images = np.mean(preact_var[j],axis=-1)
        # images = np.stack([images]*3,axis=-1)
        # save_img(images[0],"img_plots/preact_var_%i.png" % j)

# actual plotting
for i in range(img_np.shape[0]):
    save_img(img_np[i],"img_plots/i%i.png" % i)
    ent = entropy[i,...,0]
    # ent = np.maximum(entropy[i,...,0], 0)
    save_img((ent-np.min(ent))/(np.max(ent)-np.min(ent))*255,"img_plots/i%i_entropy.png" % i)
    save_img(mask[i,...,0]*255,"img_plots/i%i_mask.png" % i)
