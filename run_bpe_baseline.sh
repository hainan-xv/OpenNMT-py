#!/bin/bash

cur=$PWD

onmtdir=/export/b04/hxu/OpenNMT-py/

cd $onmtdir

#echo step-1, preprocessing

#cat $cur/train.en $cur/train.de | /export/b04/hxu/subword-nmt/learn_bpe.py -s 49500 > $cur/bpe.mdl
#cat $cur/train.en | /export/b04/hxu/subword-nmt/apply_bpe.py -c $cur/bpe.mdl > $cur/train.bpe.en
#cat $cur/train.de | /export/b04/hxu/subword-nmt/apply_bpe.py -c $cur/bpe.mdl > $cur/train.bpe.de
#
#cat $cur/dev.en | /export/b04/hxu/subword-nmt/apply_bpe.py -c $cur/bpe.mdl > $cur/dev.bpe.en
#cat $cur/dev.de | /export/b04/hxu/subword-nmt/apply_bpe.py -c $cur/bpe.mdl > $cur/dev.bpe.de
#
#python preprocess.py -train_src $cur/train.bpe.de -train_tgt $cur/train.bpe.en -valid_src $cur/dev.bpe.de -valid_tgt $cur/dev.bpe.en -save_data data/bpe
#

#python train.py -data data/bpe -save_model bpe_hainan/mdl -gpuid $(free-gpu) \
#         -rnn_size 1024 -encoder_type brnn -layers 2 -epochs 20 \
#         -dropout 0.2 -optim adadelta -learning_rate 0.4 \
#         -tgt_word_vec_size 300 -word_vec_size 300 \
#         2>&1 | tee log.train.bpe
#exit

#cat $cur/test.en | /export/b04/hxu/subword-nmt/apply_bpe.py -c $cur/bpe.mdl > $cur/test.bpe.en
#cat $cur/test.de | /export/b04/hxu/subword-nmt/apply_bpe.py -c $cur/bpe.mdl > $cur/test.bpe.de

model=`ls bpe_hainan/ | tail -n 1`

python translate.py -model bpe_hainan/$model -src $cur/test.bpe.de -output $cur/hyp.bpe.en.hainan -replace_unk -verbose -gpu $(free-gpu)

cat $cur/hyp.bpe.en.hainan | sed -r 's/(@@ )|(@@ ?$)//g' >  $cur/hyp.bpe.en.hainan.words

cat $cur/hyp.bpe.en.hainan.words | /export/b18/shuoyangd/projects/nmt_rnng/exps/exp3/steps/./.packages/mosesdecoder/VERSIONING_UNSUPPORTED/scripts/generic/multi-bleu.perl $cur/test.en

