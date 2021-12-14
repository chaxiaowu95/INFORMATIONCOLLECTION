#!/usr/bin/env bash
$model_type=$1
python freeze_graph.py --input_graph=../weights/$model_type/graph.pb \
    --input_checkpoint=.