#!/bin/bash -e

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db_main=$1
db_aux=$2
transcript=$3
data_dir=$4

# check arguments
if [ $# != 4 ]; then
    echo "Usage: $0 <db_main> <db_aux> <transcript> <data_dir>"
    exit 1
fi

# check directory existence
[ ! -e ${data_dir} ] && mkdir -p ${data_dir}

# set filenames
scp=${data_dir}/wav.scp
utt2spk=${data_dir}/utt2spk
spk2utt=${data_dir}/spk2utt
text=${data_dir}/text
segments=${data_dir}/segments

# check file existence
[ -e ${scp} ] && rm ${scp}
[ -e ${utt2spk} ] && rm ${utt2spk}
[ -e ${text} ] && rm ${text}
[ -e ${segments} ] && rm ${segments}

# clean \r at the end of each line, and thus make text
cat ${transcript} | sed 's/\r//' | sort > ${text}
echo "Successfully finished making text."

# make scp, utt2spk, and spk2utt
cat ${transcript} | cut -d ' ' -f 1 | sort | while read -r id; do
    
    if [ ${id:0:6} == "CANTTS" ]; then
        filename="${db_main}/${id}.wav"
        echo "${id} ${filename}" >> ${scp}
        echo "${id} ${id:7:2}" >> ${utt2spk}
    else
        filename="${db_aux}/${id}.wav"
        echo "${id} ${filename}" >> ${scp}
        echo "${id} LJ" >> ${utt2spk}
    fi
done
echo "Successfully finished making wav.scp, utt2spk."

utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
echo "Successfully finished making spk2utt."
