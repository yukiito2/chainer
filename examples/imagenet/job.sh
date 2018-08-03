#!/bin/sh

((iterations = 6))

echo -n "alex: "
for((n = 2304; n <= 2816; n+=128)){
	echo -n "$n: "
	python train_imagenet_data_parallel_OOC_ibm.py --arch alex -B $n -i $iterations -g 0 --ooc
}

echo -n "googlenetbn: "
for((n =  256; n <= 896; n+=128)){
	echo -n "$n: "
#	python train_imagenet_data_parallel_OOC_ibm.py --arch googlenetbn -B $n -i $iterations -g 0 --ooc
}

echo -n "resnet50: "
for((n = 64; n <= 448; n+=64)){
	echo -n "$n: "
#	python train_imagenet_data_parallel_OOC_ibm.py --arch resnet50 -B $n -i $iterations -g 0 --ooc
}
