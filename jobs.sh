#!/bin/bash
python main.py -p train -b 32 -r 16 -wm multi -nw 16 -e 71 -nch 2  -gpus 1 -pat 11
python main.py -p test -b 32 -r 16 -wm binary -nw 16 -e 71 -nch 2  -gpus 1 -pat 11
