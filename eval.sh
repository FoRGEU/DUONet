#!/bin/bash

python train_net.py --config-file configs/ShipRS_config_37+5.yaml  --eval-only \
MODEL.WEIGHTS  output/ShipRS/model_final.pth



