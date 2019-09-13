#!/bin/bash

outdir=experiments/test_cifar/
logdir=log_cifar_paper
sample_sizes="1 2 4 8 16 32 64 128 256 512"

cmd="python bitnetwork.py --dataset data.cifar10 --model.type model.small_conv --batches.size 32 --optimizer.learning_rate 5e-2 --optimizer.weight_decay 1e-5 --train_only --util.tfl tf_mod --dataset_join_train_val"

mkdir -p $outdir

# real
mode="real"
echo Doing $mode
$cmd --train_only --util.variable.transformation "real" --log.dir log_cifar_paper/real!

# 2^k*(1+m)
mode="2^k*(1+m)"
all=""
for n in $sample_sizes; do
    echo Doing $mode, n=$n
    $cmd --train_only --util.variable.transformation "2^k*(1+m)" --binom.sample_size ${n} --log.dir log_cifar_paper/2kbinom_${n}! --model.small_conv.use_bias
done

# 2^k*(1+m) (unfolded batchnorm)
mode="2^k*(1+m)"
all=""
for n in $sample_sizes; do
    echo Doing $mode, n=$n
    $cmd --train_only --util.variable.transformation "2^k*(1+m)" --binom.sample_size ${n} --log.dir log_cifar_paper/2kbinom_foldbn_${n}! --no-model.small_conv.use_bias
done
