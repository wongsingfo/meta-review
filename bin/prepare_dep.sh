#!/usr/bin/env bash

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
python -c 'import nltk; nltk.download("punkt")'
