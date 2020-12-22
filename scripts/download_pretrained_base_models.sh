#!/bin/bash

gdrive_download () {
    # Adapted from https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99

    if [ $# -ne 2 ]; then
        echo "Usage: gdrive_download gdrive_id filename"
        return 1
    fi

    gdrive_id=$1
    filename=$2

    confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=${gdrive_id}" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=${confirm}&id=${gdrive_id}" -O $2
    rm /tmp/cookies.txt
}

mkdir pretrained_base_models

for line in $(cat pretrained_base_models.csv); do
    filename=$(echo ${line} | cut -d, -f1)
    gdrive_id=$(echo ${line} | cut -d, -f2 | cut -d/ -f6)
    echo "Downloading ${filename}, ID=${gdrive_id}"
    gdrive_download ${gdrive_id} pretrained_base_models/${filename}
done
