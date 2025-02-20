#!/bin/bash

set -e

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

cd $ROOT_DIR/third_party/facefusion
python install.py --onnxruntime default

export SAPIENS_ROOT=$ROOT_DIR/third_party/sapiens
cd $SAPIENS_ROOT/engine; pip install -e .
cd $SAPIENS_ROOT/cv; pip install -e .
cd $SAPIENS_ROOT/det; pip install -e .

