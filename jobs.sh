#!/bin/bash
python main.py -ph test -b 32 -wm multi -nw 16 -e 71 -nch 1 -pat 11
python main.py -ph train -b 32 -wm binary -nw 16 -e 71 -nch 1 -pat 11
