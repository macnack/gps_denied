#!/bin/bash
DEVICE=$1
NAME=$2
FOLDER=$3
LOGFILE="eval_${DEVICE}.log"

nohup python3 eval.py train "$NAME" "$FOLDER" --device "$DEVICE" > "$LOGFILE" 2>&1 &
echo "Started training on GPU $DEVICE. Logging to $LOGFILE"
