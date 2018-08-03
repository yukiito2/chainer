#!/bin/sh

((iterations = 10))

echo "resnext101: "

((n = 1))
((h = 224))
((w = 224))
((l = 128))
echo "keep_all: "
echo "input size: $h, $w, $l"
python train_3d.py --arch resnext101 -B $n -H $h -W $w -L $l -i $iterations -g 0 --ooc --optimize_setting keep_all
echo " "
