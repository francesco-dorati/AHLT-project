#!/bin/bash

# ==========================================
# 1. Setup Log File
# ==========================================
LOG_FILE="nn_glove_results_v2.csv"

# If the log file doesn't exist, create it with headers
if [ ! -f "$LOG_FILE" ]; then
    echo "Model_Name,Batch_Size,Max_Len,Suf_Len,Epochs,Macro_Avg_F1,Status" > "$LOG_FILE"
fi

# ==========================================
# 2. Define Parameter Search Grids
# ==========================================
BATCH_SIZES=(16 32)
MAX_LENS=(100 150)
SUF_LENS=(4 5)
EPOCHS_VALS=(10 15 20)

# ==========================================
# 3. Main Tuning Loop
# ==========================================
echo "================================================="
echo " Starting Robust NN Hyperparameter Tuning"
echo "================================================="

for BS in "${BATCH_SIZES[@]}"; do
    for ML in "${MAX_LENS[@]}"; do
        for SL in "${SUF_LENS[@]}"; do
            for EP in "${EPOCHS_VALS[@]}"; do
                
                # Create a unique name for this configuration
                MODEL_NAME="NN_bs${BS}_ml${ML}_sl${SL}_ep${EP}"
                
                # ---------------------------------------------------------
                # SUPERPOWER 1: Skip Already Completed Runs
                # ---------------------------------------------------------
                # Check if this model is already logged as SUCCESS in the CSV
                if grep -q "^$MODEL_NAME,.*SUCCESS$" "$LOG_FILE"; then
                    echo "⏭️  Skipping $MODEL_NAME (Already completed successfully)"
                    continue
                fi
                
                echo "▶️  Running $MODEL_NAME (Batch=$BS, MaxLen=$ML, SufLen=$SL, Epochs=$EP)"
                
                # ---------------------------------------------------------
                # SUPERPOWER 2: Crash Resistance
                # ---------------------------------------------------------
                # Run the Python script. The 'if' statement checks if Python exited without errors.
                if python bin/run.py train predict name="$MODEL_NAME" batch_size="$BS" max_len="$ML" suf_len="$SL" epochs="$EP"; then
                    
                    # ---------------------------------------------------------
                    # SUPERPOWER 3: Safe Extraction
                    # ---------------------------------------------------------
                    STATS_FILE="results/devel-${MODEL_NAME}.stats"
                    
                    if [ -f "$STATS_FILE" ]; then
                        # Extract the Macro-average F1 score
                        SCORE=$(grep "^M\.avg[[:space:]]" "$STATS_FILE" | awk '{print $NF}')
                        
                        # Log success
                        echo "$MODEL_NAME,$BS,$ML,$SL,$EP,$SCORE,SUCCESS" >> "$LOG_FILE"
                        echo "✅ Success! Score: $SCORE"
                    else
                        # Log missing stats file
                        echo "⚠️  Warning: $STATS_FILE not found after training!"
                        echo "$MODEL_NAME,$BS,$ML,$SL,$EP,N/A,MISSING_STATS" >> "$LOG_FILE"
                    fi
                else
                    # Log crash/failure
                    echo "❌ Error: $MODEL_NAME crashed during training or prediction."
                    echo "$MODEL_NAME,$BS,$ML,$SL,$EP,N/A,FAILED" >> "$LOG_FILE"
                fi
                
                echo "-------------------------------------------------"
            done
        done
    done
done

echo "================================================="
echo " Tuning Complete! Check $LOG_FILE for full report."
echo "================================================="