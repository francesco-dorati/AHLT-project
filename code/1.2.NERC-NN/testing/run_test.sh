#!/bin/bash

# ==========================================
# FINAL TEST SET EVALUATION
# ==========================================

# 🛑 UPDATE THESE TWO VARIABLES WITH YOUR WINNER BEFORE RUNNING!
BEST_MODEL="NN_bs32_ml150_sl4_ep15"  # Example: Change this to your best model's name
BATCH_SIZE="32"                      # Change this to match your best model's batch size

echo "================================================="
echo " Preparing Test Data"
echo "================================================="
# Ensure the test.xml is parsed with the latest prefix/suffix logic
#python bin/run.py parse test

echo "================================================="
echo " Running Predictions on Unseen Test Set"
echo "================================================="
# Notice the keyword 'test' after 'predict'. 
# This forces the script to evaluate on test.pck instead of devel.pck
python bin/run.py predict test name="$BEST_MODEL" batch_size="$BATCH_SIZE"

echo "================================================="
echo " FINAL TASK 1.2 TEST SCORE FOR: $BEST_MODEL"
echo "================================================="
# Print out the final test results to the terminal
cat "results/test-${BEST_MODEL}.stats"