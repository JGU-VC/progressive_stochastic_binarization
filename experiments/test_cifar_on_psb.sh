#!/bin/bash

outdir=experiments/test_cifar/
logdir=log_cifar_paper
sample_sizes="1 2 4 8 16 32 64 128 256 512"

cmd="python bitnetwork.py --dataset data.cifar10 --model.type model.small_conv --batches.size 32 --optimizer.learning_rate 5e-2 --optimizer.weight_decay 1e-5 --test_only --util.tfl tf_mod"

mkdir -p $outdir

# real
mode="real"
echo Doing $mode
all=""
for n in $sample_sizes; do
    echo Doing $mode, n=$n
    res=$($cmd --util.variable.transformation "2^k*(1+m)" --binom.sample_size ${n} --log.dir log_cifar_paper/real! | tail -n 1)
    all="$all$modelname,$mode,$m,$res\n"
done
printf "$all" > $outdir/real_${mode}.csv


# 2^k*(1+m)
mode="2^k*(1+m)_nofold"
all=""
for n in $sample_sizes; do
    echo Doing $mode, n=$n
    all=""
    for m in $sample_sizes; do
        echo Doing $mode, n=$n
        res=$($cmd --util.variable.transformation "2^k*(1+m)" --binom.sample_size ${m} --log.dir log_cifar_paper/2kbinom_${n}! --model.small_conv.use_bias | tail -n 1)
        all="$all$modelname,$mode,$m,$res\n"
    done
    printf "$all" > $outdir/2kbinom_${n}_${mode}.csv
done

# # 2^k*(1+m)
mode="2^k*(1+m)_fold"
all=""
for n in $sample_sizes; do
    echo Doing $mode, n=$n
    all=""
    for m in $sample_sizes; do
        echo Doing $mode, n=$n
        res=$($cmd --util.variable.transformation "2^k*(1+m)" --binom.sample_size ${m} --log.dir log_cifar_paper/2kbinom_foldbn_${n}! --no-model.small_conv.use_bias | tail -n 1)
        all="$all$modelname,$mode,$m,$res\n"
    done
    printf "$all" > $outdir/2kbinom_foldbn_${n}_${mode}.csv
done
