IMPROVE_MODEL_NAME=deepttc
IMPROVE_MODEL_SCRIPT=${IMPROVE_MODEL_NAME}_infer_improve.py

# Set env if CANDLE_MODEL is not in same directory as this script
IMPROVE_MODEL_DIR=${IMPROVE_MODEL_DIR:-$( dirname -- "$0" )}

python $IMPROVE_MODEL_DIR/$IMPROVE_MODEL_SCRIPT $@
