#!/bin/bash
python main.py -p train -b 64 -r 16 -wm multi_reg -nw 24 -e 51
python main.py -p train -b 64 -r 16 -wm binary -nw 24 -e 51
python main.py -p train -b 64 -r 16 -wm multi -nw 24 -e 51
python main.py -p train -b 64 -r 16 -wm multi_reg_random -nw 24 -e 51