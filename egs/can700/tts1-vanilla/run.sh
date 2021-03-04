#!/bin/bash

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1
stop_stage=100
ngpu=1       # number of gpus ("0" uses cpu, otherwise use gpu)
nj=32        # number of parallel jobs
dumpdir=dump # directory to dump full features
verbose=1    # verbose option (if set > 0, get more log)
N=0          # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)

# feature extraction related
fs=22050        # sampling frequency
fmax=7600       # maximum frequency
fmin=80         # minimum frequency
n_mels=80       # number of mel basis
n_fft=1024      # number of fft points
n_shift=256     # number of shift points
win_length=1024 # window length

# config files
train_config=conf/train_pytorch_tacotron2.yaml
decode_config=conf/decode.yaml

# decoding related
model=model.loss.best
n_average=1 # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# feature related
use_intotype=false
use_charembed=false
use_spkid=false

# root directory of db
data_dir=/data1/baibing/datasets/CantoneseTTS/CANTTSdata_22050Hz
lj_data_dir=/data1/baibing/datasets/LJSpeech-1.1/wavs
char_emb_dir=/data1/baibing/datasets/CantoneseTTS/text_embeddings
cmvn_path=/data1/baibing/datasets/CantoneseTTS/cmvn_lj_pwg/cmvn.ark
master_port=29507

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_file=`which "tts_train.py"`

train_set="train_no_dev"
train_dev="dev"
eval_set="eval"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    for dir in ${train_set} ${train_dev} ${eval_set}; do
        local/data_prep.sh ${data_dir} ${lj_data_dir} local/can_${dir} data/${dir}
        utils/validate_data_dir.sh --no-feats data/${dir}
    done
fi

feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}; mkdir -p ${feat_dt_dir}
feat_ev_dir=${dumpdir}/${eval_set}; mkdir -p ${feat_ev_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"

    # Generate the fbank features; by default 80-dimensional fbanks on each frame
    fbankdir=fbank
    for dir in ${train_set} ${train_dev} ${eval_set}; do
        make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --n_mels ${n_mels} \
            data/${dir} \
            exp/make_fbank/${dir} \
            ${fbankdir}
    done

    # remove utterances that are too long
    mv data/${train_set} data/${train_set}_org
    mv data/${train_dev} data/${train_dev}_org
    remove_longshortdata.sh --maxframes 2500 --maxchars 330 data/${train_set}_org data/${train_set}
    remove_longshortdata.sh --maxframes 2500 --maxchars 330 data/${train_dev}_org data/${train_dev}

    # compute statistics for global mean-variance normalization
    #compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark
    cp ${cmvn_path} data/${train_set}/cmvn.ark
    echo "Copied pre-calculated cmvn."

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${eval_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/eval ${feat_ev_dir}
fi

dict=data/lang_phn/${train_set}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_phn/
    # if <unk> does not exist in the transcription, add it and change "NR" to "NR+1".
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for padding idx
    text2token.py -s 1 -n 1 --trans_type phn data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp --trans_type phn \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --trans_type phn \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    data2json.sh --feat ${feat_ev_dir}/feats.scp --trans_type phn \
         data/${eval_set} ${dict} > ${feat_ev_dir}/data.json
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Adding extra features to data.json"
    # update json
    for name in ${train_set} ${train_dev} ${eval_set}; do
        # to store intermediate json files
        tmpdir=${dumpdir}/${name}/tmpdir
        rm -rf ${tmpdir} && mkdir ${tmpdir}
        [ ! -d ${dumpdir}/${name}/.backup ] && mkdir ${dumpdir}/${name}/.backup
        # backup
        cp ${dumpdir}/${name}/data.json ${dumpdir}/${name}/.backup/data.json
        cp ${dumpdir}/${name}/data.json ${tmpdir}/data.json
        # iteratively update the json file
        # add character embeddings to output
        if ${use_charembed}; then
            cat data/${name}/wav.scp | awk -F' ' -v dir=$char_emb_dir '{printf "%s %s/%s.npy\n",$1,dir,$1}' > data/${name}/char_emb.scp
            cat data/${name}/char_emb.scp | scp2json.py --key char_emb > ${tmpdir}/char_emb.json
            addjson.py -i False \
                ${tmpdir}/${name}/data.json \
                ${tmpdir}/char_emb.json \
                > ${tmpdir}/data.json
        fi
        # add intonation types to output
        if ${use_intotype}; then
            cat data/${name}/wav.scp \
                | awk -F' ' '{type=0; if(match($1,"FQ")) type=1; if(match($1,"FU")) type=2; printf "%s %s\n",$1,type}' \
                > data/${name}/into_type.scp
            cat data/${name}/into_type.scp | scp2json.py --key into_type > ${tmpdir}/into_type.json
            addjson.py -i False \
                ${tmpdir}/data.json \
                ${tmpdir}/into_type.json \
                > ${dumpdir}/${name}/data.json
        fi
        # add speaker ids to output
        if ${use_spkid}; then
            cat data/${name}/wav.scp \
                | awk -F' ' '{spk=0; if(match($1,"LJ")) spk=1; printf "%s %s\n",$1,spk}' \
                > data/${name}/spkid
            cat data/${name}/spkid | scp2json.py --key spkid > ${tmpdir}/spkid.json
            addjson.py -i False \
                ${tmpdir}/data.json \
                ${tmpdir}/spkid.json \
                > ${dumpdir}/${name}/data.json
        fi
        # remove temporary files
        rm -rf ${tmpdir}
    done
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Text-to-speech model training"
    tr_json=${feat_tr_dir}/data.json
    dt_json=${feat_dt_dir}/data.json
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        python -m torch.distributed.launch \
            --nproc_per_node ${ngpu} \
            --master_port ${master_port} \
            ${train_file} \
                --backend ${backend} \
                --ngpu ${ngpu} \
                --minibatches ${N} \
                --outdir ${expdir}/results \
                --tensorboard-dir tensorboard/${expname} \
                --verbose ${verbose} \
                --seed ${seed} \
                --resume ${resume} \
                --train-json ${tr_json} \
                --valid-json ${dt_json} \
                --config ${train_config}
fi

if [ ${n_average} -gt 0 ]; then
    model=model.last${n_average}.avg.best
fi
outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    if [ ${n_average} -gt 0 ]; then
        average_checkpoints.py --backend ${backend} \
                               --snapshots ${expdir}/results/snapshot.ep.* \
                               --out ${expdir}/results/${model} \
                               --num ${n_average}
    fi
    pids=() # initialize pids
    for sets in ${train_dev} ${eval_set}; do
    (
        [ ! -e ${outdir}/${sets} ] && mkdir -p ${outdir}/${sets}
        cp ${dumpdir}/${sets}/data.json ${outdir}/${sets}
        splitjson.py --parts ${nj} ${outdir}/${sets}/data.json
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${sets}/log/decode.JOB.log \
            tts_decode.py \
                --backend ${backend} \
                --ngpu 0 \
                --verbose ${verbose} \
                --out ${outdir}/${sets}/feats.JOB \
                --json ${outdir}/${sets}/split${nj}utt/data.JOB.json \
                --model ${expdir}/results/${model} \
                --config ${decode_config}
        # concatenate scp files
        for n in $(seq ${nj}); do
            cat "${outdir}/${sets}/feats.$n.scp" || exit 1;
        done > ${outdir}/${sets}/feats.scp
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Synthesis"
    pids=() # initialize pids
    for sets in ${train_dev} ${eval_set}; do
    (
        [ ! -e ${outdir}_denorm/${sets} ] && mkdir -p ${outdir}_denorm/${sets}
        apply-cmvn --norm-vars=true --reverse=true data/${train_set}/cmvn.ark \
            scp:${outdir}/${sets}/feats.scp \
            ark,scp:${outdir}_denorm/${sets}/feats.ark,${outdir}_denorm/${sets}/feats.scp
        convert_fbank.sh --nj ${nj} --cmd "${train_cmd}" \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --n_mels ${n_mels} \
            --iters ${griffin_lim_iters} \
            ${outdir}_denorm/${sets} \
            ${outdir}_denorm/${sets}/log \
            ${outdir}_denorm/${sets}/wav
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished."
fi
