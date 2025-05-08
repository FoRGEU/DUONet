#!/bin/bash

python /home/quchenyu/DUONet/train_net.py --config-file /home/quchenyu/DUONet/configs/ShipRS_config_37+5.yaml  --eval-only \
MODEL.WEIGHTS  /home/quchenyu/DUONet/output/ShipRS/model_final.pth



