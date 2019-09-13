#!/bin/bash
module add lib/TensorFlow/1.13.1-fosscuda-2019.01-Python-3.5.6 


outdir=experiments/test_attention_resnet18_full


params="--test_only --no-log.summaries --model.type model.tf_resnet_official --batches.size 64 --attention_predict --no-log.optimistic_restore --log.dir ckpts_imgn/resnet18_slim!"
# add_params="$add_params --test_subset 0.01"
add_params="$add_params"
sample_sizes="1 2 4 8 16 32"

mkdir -p $outdir


attention_params_list=(
    "--attention.mode spatial --attention.spatial_mode random --attention.fraction 0.336978"
    "--attention.mode spatial --attention.spatial_mode random --attention.fraction 0.762942"
    "--attention.mode spatial --attention.spatial_mode mean_entropy --attention.spatial_surround 3 --attention.fraction 1"
    "--attention.mode spatial --attention.spatial_mode mean_entropy --attention.spatial_surround 0 --attention.fraction 1"
)


mode="psb"
# GPU=$(printenv CUDA_VISIBLE_DEVICES)
GPU=$(printenv DEVICE_NUM)
GPUNUM=$(printenv CUDA_NUM_DEVICES)
for i in "${!attention_params_list[@]}"; do
    attention_params="${attention_params_list[$i]}"
    exp_name=$(echo $attention_params | sed -r 's/_//g' | sed -r 's/\s*--\w+(\.(\w+))*\s+([0-9]+)/_\2\3/g' | sed -r 's/\s*--\w+(\.\w+)*\s+/_/g')
    if [ ! -z $GPU ] && [ $(($i % $GPUNUM == $GPU)) != "1" ]; then
        continue;
    fi
    echo $exp_name
    mkdir -p $outdir/$exp_name
    for n in $sample_sizes; do
        for m in $sample_sizes; do
            if [ $n -gt $m ]; then
                continue;
            fi

            mode_params="$attention_params --util.variable.transformation psb  --binom.sample_size $n
            --attention.transform psb --attention.sample_size $m"
            if [ ! -f $outdir/$exp_name/${mode}_${n}_${m}_acc.csv ]; then
                echo Doing $mode $n $m
                # echo python bitnetwork.py $params $add_params $mode_params
                res=$(python bitnetwork.py $params $add_params $mode_params)
                printf "$res" > $outdir/$exp_name/${mode}_${n}_${m}.csv
                res_acc=$(cat $outdir/$exp_name/${mode}_${n}_${m}.csv | tail -n 1)
                res_prop=$(cat $outdir/$exp_name/${mode}_${n}_${m}.csv | tail -n 2 | head -n 1)
                printf "$res_acc" > $outdir/$exp_name/${mode}_${n}_${m}_acc.csv
                printf "$res_prop" > $outdir/$exp_name/${mode}_${n}_${m}_prop.csv
            fi
        done
    done
done



