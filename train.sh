python modeling/train.py \
  --batch-size=2048 \
  --epochs=5000 \
  --path=models/conv_mcts_10m.pt \
  --existing-model=models/conv_mcts_10m.pt \
  --data=datasets/mcts_10m_aug.h5 \
  --num-workers=16 \
  --learning-rate=0.0015

    #> output.log 2>&1 &

python modeling/train.py \
  --batch-size=2048 \
  --epochs=5000 \
  --path=models/mcts_50m.pt \
  --data=datasets/mcts_50m.h5 \
  --num-workers=16 \
  --learning-rate=0.0015
