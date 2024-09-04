#!/bin/bash

SCRIPT_NAME="main_button.py"

SAVE_PATH=/home/yzq/hzy/DOIC/Results

SIGNAL_FILE="$SAVE_PATH/temp.txt"

PID=$(pgrep -f "$SCRIPT_NAME")

if [ -f "$SIGNAL_FILE" ]; then
    echo "Stopping Program(PID: $PID) ..."
    kill -SIGINT "$PID"
    rm "$SAVE_PATH/temp.txt"
else
    echo "Start Program ..." >> $SIGNAL_FILE
fi
		
