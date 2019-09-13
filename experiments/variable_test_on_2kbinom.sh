#!/bin/bash

dir=log_cifar_traintestdiff
testn="1 2 4 8 16 24 32 48 56 64 72 98 112 128 256 512"

if [ ! -f "$dir/real" ]; then
	python bitnetwork.py --train_only --util.variable.transformation 'real' --log.dir $dir/real!
fi
if [ ! -f "$dir/2^k*p" ]; then
	python bitnetwork.py --train_only --util.variable.transformation '2^k*p' --log.dir $dir/2kreal!
fi

python py/train_n_test_m.py --train real 2kreal --test $test_n --log.dir "$dir/" --cmd python bitnetwork.py --util.variable.transformation '2^k*binom_p' --binom.sample_size {1}
python py/train_n_test_m.py --train 1 2 4 8 16 32 64 128 256 512 --test $testn --log.dir "$dir/2kbinom_" --cmd python bitnetwork.py --util.variable.transformation '2^k*binom_p' --binom.sample_size {1}

