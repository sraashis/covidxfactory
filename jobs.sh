#!/bin/bash
python main.py -p train -b 64 -r 16 -wm multi_reg -nw 24 -e 101
python main.py -p train -b 64 -r 16 -wm binary -nw 24 -e 101
python main.py -p train -b 64 -r 16 -wm multi -nw 24 -e 101