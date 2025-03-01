python modeling/evaluation/train.py \
  --batch-size=4096 \
  --epochs=5000 \
  --path=models/eval_conv_1m.pt \
  --data=datasets/positions_1m.h5 \
  --num-workers=16 \
  --learning-rate=0.002 \
  --min-learning-rate=0.000001 > output.log 2>&1 &
