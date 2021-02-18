#!/bin/bash

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;

# synth_model="exp/train_no_dev_pytorch_train_pytorch_transformer/results/model.last1.avg.best"
synth_model="exp/train_no_dev_pytorch_train_pytorch_tacotron2/results/model.last1.avg.best"
# vocoder="ljspeech_parallel_wavegan.v1"
vocoder="can700_multi_band_melgan.v2"

decode_dir="decode"
dict="data/lang_phn/train_no_dev_units.txt"
cmvn="data/train_no_dev/cmvn.ark"
decode_config="conf/decode.yaml"
vocoder_dir="/data1/baibing/.cache/parallel_wavegan"

fs=22050      # sampling frequency
fmax=7600       # maximum frequency
fmin=80       # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024    # number of fft points
n_shift=256   # number of shift points
win_length=1024 # window length
griffin_lim_iters=64

verbose=1
nj=1
ngpu=1
trans_type="phn"

stage=-1
stop_stage=100
text=""

help_message=$(cat <<EOF
Usage:
    $ $0 --text should be given.
EOF
)

. utils/parse_options.sh || exit 1

if [ -z vocoder_dir ]; then
    vocoder_dir=${decode_dir}/download
fi

if [ -z ${text} ]; then
    echo "${help_message}"
    exit 1;
fi

set -e
set -u
set -o pipefail

function download_vocoders () {
    case "${vocoder}" in
        "ljspeech.wavenet.softmax.ns.v1") share_url="https://drive.google.com/open?id=1eA1VcRS9jzFa-DovyTgJLQ_jmwOLIi8L";;
        "ljspeech.wavenet.mol.v1") share_url="https://drive.google.com/open?id=1sY7gEUg39QaO1szuN62-Llst9TrFno2t";;
        "ljspeech.parallel_wavegan.v1") share_url="https://drive.google.com/open?id=1tv9GKyRT4CDsvUWKwH3s_OfXkiTi0gw7";;
        "libritts.wavenet.mol.v1") share_url="https://drive.google.com/open?id=1jHUUmQFjWiQGyDd7ZeiCThSjjpbF_B4h";;
        "jsut.wavenet.mol.v1") share_url="https://drive.google.com/open?id=187xvyNbmJVZ0EZ1XHCdyjZHTXK9EcfkK";;
        "jsut.parallel_wavegan.v1") share_url="https://drive.google.com/open?id=1OwrUQzAmvjj1x9cDhnZPp6dqtsEqGEJM";;
        "csmsc.wavenet.mol.v1") share_url="https://drive.google.com/open?id=1PsjFRV5eUP0HHwBaRYya9smKy5ghXKzj";;
        "csmsc.parallel_wavegan.v1") share_url="https://drive.google.com/open?id=10M6H88jEUGbRWBmU1Ff2VaTmOAeL8CEy";;
        *) echo "No such models: ${vocoder}"; exit 1 ;;
    esac

    dir=${vocoder_dir}/${vocoder}
    mkdir -p "${dir}"
    if [ ! -e "${dir}/.complete" ]; then
        download_from_google_drive.sh "${share_url}" "${dir}" ".tar.gz"
	touch "${dir}/.complete"
    fi
}

phn=${decode_dir}/phn
phn_ids=${decode_dir}/phn_ids

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: converting text to phonemes..."
    [ ! -e ${decode_dir} ] && mkdir -p ${decode_dir}

    local/g2p.py ${text} ${phn} --add_id "False" --sos "<sos>"
    # local/g2p.py ${text} ${phn} --add_id "False"
fi

base=$(basename "${phn}" .txt)

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: preparing data in json..."
    [ -e "${decode_dir}/data" ] && rm -rf "${decode_dir}/data"
    mkdir -p "${decode_dir}/data"
    num_lines=$(wc -l < "${phn}")
    for idx in $(seq "${num_lines}"); do
        echo "${base}_${idx} X" >> "${decode_dir}/data/wav.scp"
        echo "X ${base}_${idx}" >> "${decode_dir}/data/spk2utt"
        echo "${base}_${idx} X" >> "${decode_dir}/data/utt2spk"
        echo -n "${base}_${idx} " >> "${decode_dir}/data/text"
        sed -n "${idx}"p "${phn}" >> "${decode_dir}/data/text"
    done
    mkdir -p "${decode_dir}/dump"
    data2json.sh --trans_type "${trans_type}" "${decode_dir}/data" "${dict}" > "${decode_dir}/dump/data.json"
fi

# Need: decode_config synth_model

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: synthesizing spectrograms..."
    ${decode_cmd} "${decode_dir}/log/decode.log" \
        tts_decode.py \
        --config "${decode_config}" \
        --ngpu "${ngpu}" \
        --backend pytorch \
        --verbose "${verbose}" \
        --out "${decode_dir}/outputs/feats" \
        --json "${decode_dir}/dump/data.json" \
        --model "${synth_model}"
fi

outdir=${decode_dir}/outputs; mkdir -p "${outdir}_denorm"

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: synthesizing waveform with Griffin-Lim..."
    apply-cmvn --norm-vars=true --reverse=true "${cmvn}" \
        scp:"${outdir}/feats.scp" \
        ark,scp:"${outdir}_denorm/feats.ark,${outdir}_denorm/feats.scp"
    echo "applied denormalization on synthesized features."
    convert_fbank.sh --nj 1 --cmd "${decode_cmd}" \
        --fs "${fs}" \
        --fmax "${fmax}" \
        --fmin "${fmin}" \
        --n_fft "${n_fft}" \
        --n_shift "${n_shift}" \
        --win_length "${win_length}" \
        --n_mels "${n_mels}" \
        --iters "${griffin_lim_iters}" \
        "${outdir}_denorm" \
        "${decode_dir}/log" \
        "${decode_dir}/wav_gl"
    echo "speech synthesized with Griffin-Lim in ${decode_dir}/wav_gl"
fi

dst_dir=${decode_dir}/wav_nn

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: synthesizing waveform with a neural vocoder..."
    echo "using vocoder (${vocoder}) without checking sampling rate."
    if [[ "${vocoder}" == *"parallel_wavegan."* || "${vocoder}" == *"multi_band_melgan."* ]]; then
        checkpoint=$(find "${vocoder_dir}/${vocoder}" -name "*.pkl" | head -n 1)
        if [ -z "${checkpoint}" ]; then
            echo "Vocoder not downloaded. Trying downloading it..."
            download_vocoders
        fi
        if ! command -v parallel-wavegan-decode > /dev/null; then
            pip install parallel-wavegan
        fi
        parallel-wavegan-decode \
            --scp "${outdir}/feats.scp" \
            --checkpoint "${checkpoint}" \
            --outdir "${dst_dir}" \
            --verbose "${verbose}"
    else
        echo "vocoder not supported!"
        echo "exiting."
        exit 1
    fi
    echo "speech synthesized with Griffin-Lim in ${decode_dir}/wav_nn"
fi

echo "All finished."
