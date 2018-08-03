#!/bin/sh

((iterations = 10))

echo "googlenetbn: "

((n = 256))
echo "keep_all: "
echo "bacth size: $n"
python train_imagenet_data_parallel_OOC_ibm.py --arch googlenetbn -B $n -i $iterations -g 0 --ooc --optimize_setting keep_all
echo " "

echo "pooch: "
for((n = 256; n <= 896; n+=128)){
	echo "bacth size: $n"
	python train_imagenet_data_parallel_OOC_ibm.py --arch googlenetbn -B $n -i $iterations -g 0 --ooc
	echo " "
}

echo "superneurons: "
for((n = 256; n <= 896; n+=128)){
	echo "bacth size: $n"
	python train_imagenet_data_parallel_OOC_ibm.py --arch googlenetbn -B $n -i $iterations -g 0 --ooc --optimize_setting superneurons
	echo " "
}