INPUT_PATH=$1
OUTPUT_PATH=$2
MODEL_PATH=$3

python transform_video.py --in-path $INPUT_PATH \
  --checkpoint $MODEL_PATH \
  --out-path $OUTPUT_PATH \
  --device /gpu:0 \
  --batch-size 1