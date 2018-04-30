#!/bin/bash

stage=3

onmtdir=/export/b05/hxu/OpenNMT-py
datadir=/export/b05/hxu/MT-data
dir=$datadir/bpe

cd $onmtdir
mkdir -p $dir/

if [ $stage -le 1 ]; then
  echo step-1, preprocessing
  cat $datadir/train.en $datadir/train.de | /export/b05/hxu/subword-nmt/learn_bpe.py -s 49500        > $dir/bpe.mdl

  cat $datadir/train.en                   | /export/b04/hxu/subword-nmt/apply_bpe.py -c $dir/bpe.mdl > $dir/train.en
  cat $datadir/train.de                   | /export/b04/hxu/subword-nmt/apply_bpe.py -c $dir/bpe.mdl > $dir/train.de
  cat $datadir/dev.en                     | /export/b04/hxu/subword-nmt/apply_bpe.py -c $dir/bpe.mdl > $dir/dev.en
  cat $datadir/dev.de                     | /export/b04/hxu/subword-nmt/apply_bpe.py -c $dir/bpe.mdl > $dir/dev.de
  cat $datadir/test.en                    | /export/b04/hxu/subword-nmt/apply_bpe.py -c $dir/bpe.mdl > $dir/test.en
  cat $datadir/test.de                    | /export/b04/hxu/subword-nmt/apply_bpe.py -c $dir/bpe.mdl > $dir/test.de
  exit

fi


if [ $stage -le 2 ]; then
  echo step-2, training
  python preprocess.py -train_src $dir/train.de -train_tgt $dir/train.en -valid_src $dir/dev.de -valid_tgt $dir/dev.en -save_data $dir/onmt_data
  python train.py -data $dir/onmt_data -save_model $dir/model/onmt -gpuid $(free-gpu) \
                  -rnn_size 1024 -encoder_type brnn -layers 2 -epochs 20 \
                  -dropout 0.2 -optim adadelta -learning_rate 1.0 \
                  -tgt_word_vec_size 300 -word_vec_size 300 \
                  2>&1 | tee $dir/log_train
  exit
fi

if [ $stage -le 3 ]; then
  model=`ls $dir/model/ | tail -n 1`
  echo Using $model

#  python translate.py -model $dir/model/$model -src $dir/test.de -output $dir/hyp.bpe.en -replace_unk -verbose -gpu $(free-gpu)
  cat $dir/hyp.bpe.en | sed -r 's/(@@ )|(@@ ?$)//g' >  $dir/test.en
  cat $dir/hyp.en | /export/b18/shuoyangd/projects/nmt_rnng/exps/exp3/steps/./.packages/mosesdecoder/VERSIONING_UNSUPPORTED/scripts/generic/multi-bleu.perl -lc $datadir/test.en
fi
sleep 1m
if [ $stage -le 4 ]; then
  model=`ls $dir/model/ | tail -n 1`
  echo Using $model

  cat $datadir/train.de | sed "s= =\n=g" | sort | uniq -c | sort -k1nr | head -n 3000 | awk '{print $2}' > $dir/top_words.de
  cat $dir/top_words.de | /export/b04/hxu/subword-nmt/apply_bpe.py -c $dir/bpe.mdl > $dir/top_words.bpe.de

  python translate.py -model $dir/model/$model -src $dir/top_words.bpe.de -output $dir/top_words.bpe.en -replace_unk -verbose -gpu $(free-gpu)
  cat $dir/top_words.bpe.en | sed -r 's/(@@ )|(@@ ?$)//g' >  $dir/top_words.en
#  cat $dir/hyp.en | /export/b18/shuoyangd/projects/nmt_rnng/exps/exp3/steps/./.packages/mosesdecoder/VERSIONING_UNSUPPORTED/scripts/generic/multi-bleu.perl $datadir/test.en
fi
