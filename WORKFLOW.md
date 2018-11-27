# Training on SQUAD:

git clone https://github.com/huggingface/pytorch-pretrained-BERT
cd pytorch-pretrained-BERT
python3.6 -m venv env
. env/bin/activate
mkdir models squad output
cd squad
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
wget https://github.com/allenai/bi-att-flow/raw/master/squad/evaluate-v1.1.py
cd ..
pip install -U numpy tensorflow
pip install -r requirements.txt
export SQUAD_DIR=./squad
export OUTPUT_DIR=./output
export PYTORCH_PRETRAINED_BERT_CACHE=./models


```
python examples/run_squad.py \
  --bert_model bert-base-cased \
  --do_train \
  --do_predict \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --train_batch_size 18 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $OUTPUT_DIR
```

```
python squad/evaluate-v1.1.py squad/dev-v1.1.json output/predictions.json 

```
