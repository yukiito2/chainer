#!/bin/sh

((iterations = 10))

echo "resnext101: "

((n = 1))

echo "keep_all: "
((h = 224))
((w = 224))
((l = 128))
echo "input size: $h, $w, $l"
python train_3d.py --arch resnext101 -B $n -H $h -W $w -L $l -i $iterations -g 0 --ooc --optimize_setting keep_all
echo " "

echo "pooch: "
((h = 224))
((w = 224))
((l = 128))
echo "input size: $h, $w, $l"
python train_3d.py --arch resnext101 -B $n -H $h -W $w -L $l -i $iterations -g 0 --ooc
echo " "

((h = 224))
((w = 224))
((l = 256))
echo "input size: $h, $w, $l"
python train_3d.py --arch resnext101 -B $n -H $h -W $w -L $l -i $iterations -g 0 --ooc
echo " "

((h = 448))
((w = 224))
((l = 128))
echo "input size: $h, $w, $l"
python train_3d.py --arch resnext101 -B $n -H $h -W $w -L $l -i $iterations -g 0 --ooc
echo " "

((h = 448))
((w = 224))
((l = 256))
echo "input size: $h, $w, $l"
python train_3d.py --arch resnext101 -B $n -H $h -W $w -L $l -i $iterations -g 0 --ooc
echo " "


echo "superneurons: "
((h = 224))
((w = 224))
((l = 128))
echo "input size: $h, $w, $l"
python train_3d.py --arch resnext101 -B $n -H $h -W $w -L $l -i $iterations -g 0 --ooc --optimize_setting superneurons
echo " "

((h = 224))
((w = 224))
((l = 256))
echo "input size: $h, $w, $l"
python train_3d.py --arch resnext101 -B $n -H $h -W $w -L $l -i $iterations -g 0 --ooc --optimize_setting superneurons
echo " "

((h = 448))
((w = 224))
((l = 128))
echo "input size: $h, $w, $l"
python train_3d.py --arch resnext101 -B $n -H $h -W $w -L $l -i $iterations -g 0 --ooc --optimize_setting superneurons
echo " "

((h = 448))
((w = 224))
((l = 256))
echo "input size: $h, $w, $l"
python train_3d.py --arch resnext101 -B $n -H $h -W $w -L $l -i $iterations -g 0 --ooc --optimize_setting superneurons
echo " "