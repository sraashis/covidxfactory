#!/bin/bash
python main.py -p train -b 32 -r 16 -wm binary -nw 16 -e 51 -nch 2
python main.py -p train -b 32 -r 16 -wm multi -nw 16 -e 51 -nch 2
python main.py -p train -b 32 -r 16 -wm multi_reg -nw 16 -e 51 -nch 2
cd