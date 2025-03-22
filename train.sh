python modeling/train.py \
  --batch-size=2048 \
  --epochs=5000 \
  --path=models/conv_pos_mixed.pt \
  --data=datasets/mixed_23m.h5 \
  --num-workers=16 \
  --learning-rate=0.0015  > output.log 2>&1 &
