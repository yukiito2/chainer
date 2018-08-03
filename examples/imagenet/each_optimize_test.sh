#!/bin/sh

((iterations = 10))

echo "resnet50: "
((n = 256))
echo "swap_all_no_scheduling"
python train_imagenet_data_parallel_OOC_ibm.py --arch resnet50 -B $n -i $iterations -g 0 --ooc --optimize_setting swap_all_no_scheduling
echo "swap_all"
python train_imagenet_data_parallel_OOC_ibm.py --arch resnet50 -B $n -i $iterations -g 0 --ooc --optimize_setting swap_all
echo "swap_opt"
python train_imagenet_data_parallel_OOC_ibm.py --arch resnet50 -B $n -i $iterations -g 0 --ooc --optimize_setting swap_opt
echo "pooch"
python train_imagenet_data_parallel_OOC_ibm.py --arch resnet50 -B $n -i $iterations -g 0 --ooc

echo ""
echo "googlenetbn: "
((n = 384))
echo "swap_all_no_scheduling"
python train_imagenet_data_parallel_OOC_ibm.py --arch googlenetbn -B $n -i $iterations -g 0 --ooc --optimize_setting swap_all_no_scheduling
echo "swap_all"
python train_imagenet_data_parallel_OOC_ibm.py --arch googlenetbn -B $n -i $iterations -g 0 --ooc --optimize_setting swap_all
echo "swap_opt"
python train_imagenet_data_parallel_OOC_ibm.py --arch googlenetbn -B $n -i $iterations -g 0 --ooc --optimize_setting swap_opt
echo "pooch"
python train_imagenet_data_parallel_OOC_ibm.py --arch googlenetbn -B $n -i $iterations -g 0 --ooc

echo ""
echo "alex: "
((n = 2432))
echo "swap_all_no_scheduling"
python train_imagenet_data_parallel_OOC_ibm.py --arch alex -B $n -i $iterations -g 0 --ooc --optimize_setting swap_all_no_scheduling
echo "swap_all"
python train_imagenet_data_parallel_OOC_ibm.py --arch alex -B $n -i $iterations -g 0 --ooc --optimize_setting swap_all
echo "swap_opt"
python train_imagenet_data_parallel_OOC_ibm.py --arch alex -B $n -i $iterations -g 0 --ooc --optimize_setting swap_opt
echo "pooch"
python train_imagenet_data_parallel_OOC_ibm.py --arch alex -B $n -i $iterations -g 0 --ooc