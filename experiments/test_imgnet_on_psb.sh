#!/bin/bash

outdir=experiments/test_imagenet/

models="nasnetmobile mobilenet resnet50 resnet50v2 xception inceptionresnetv2 densenet121 inceptionv3"

add_params="--test_only --no-log.summaries"
# add_params="$add_params --test_subset 0.0001"
sample_sizes="1 2 4 8 16 32"

mkdir -p $outdir

for modelname in $models; do

    # ensure model is available
    echo "Preparing " $modelname
    python py/download_and_convert_keras_model.py $modelname;
    if [ $? -eq 1 ]; then
        echo "Download Succeeded"
    else
        echo "ERROR" 
        exit
    fi

    # evaluate real
    mode="real"
    echo Doing $mode
    res=$(python bitnetwork.py $add_params --util.variable.transformation "real" --model.type model.classification_models --model.classification_models.model $modelname --batches.test_batch_size 64 | tail -n 1)
    all="$modelname,real,0,$res\n"
    printf "$all" > $outdir/${modelname}_real.csv

    # 2^k*(1+m)
    mode="2^k*(1+m)"
    all=""
    for n in $sample_sizes; do
        echo Doing $mode, n=$n
        # res=$n
        res=$(python bitnetwork.py $add_params --util.variable.transformation "2^k*(1+m)" --model.type model.classification_models --model.classification_models.model $modelname --binom.sample_size $n --batches.test_batch_size 32 | tail -n 1)
        all="$all$modelname,$mode,$n,$res\n"
        printf "$modelname,$mode,$n,$res" > $outdir/${modelname}_${mode}_subresult_n${n}.csv
    done
    printf "$all" > $outdir/${modelname}_${mode}.csv


    # fp-only
    mode="fp-only"
    all=""
    for n in $sample_sizes; do
        # res=$n
        res=$(python bitnetwork.py $add_params --util.variable.transformation "real" --model.type model.classification_models --model.classification_models.model $modelname --util.variable.fixed_point.bits $n --batches.test_batch_size 32 | tail -n 1)
        all="$all$modelname,$mode,$n,$res\n"
    done
    printf "$all" > $outdir/${modelname}_$mode.csv

done


