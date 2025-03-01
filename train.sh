python modeling/train.py \
  --batch-size=4096 \
  --epochs=5000 \
  --path=models/convolutional_positions_1m_aug.pt \
  --data=datasets/positions_1m_aug.h5 \
  --num-workers=16 \
  --learning-rate=0.001  > output.log 2>&1 &
