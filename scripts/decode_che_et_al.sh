#!/bin/bash
# This scripts takes the sentences .lines file and the annotated .conllu file of a dataset
# and generates the semantic graphs .rdf and associated metadata .metadata
# Assumes the presence of ${DATASET_DIR}/{train,dev,test}.{lines,conllu}
# Make sure the Python environment has
# - allennlp==0.9.0 (note this is different from our main requirement!)
# - an appropriate cuda-enabled PyTorch version
# - rdflib
# And $CUDA_VISIBLE_DEVICES is set

set -e

if [[ ${#} -ne 1 ]]; then
  echo "usage: scripts/decode_che_et_al.sh DATASET_DIR"
  exit 1;
fi

PROJECT_DIR=$(pwd)
DATASET_DIR=$(cd $1; pwd)

echo ${DATASET_DIR}

N_VISIBLE_GPUS=`nvidia-smi -L | wc -l`

for SPLIT in train dev test; do
    echo "Working on split ${SPLIT}"

    cd ${PROJECT_DIR}/..
    echo "Converting to UDPipe"
    # The resulting file is equivalent to the `evaluation/udpipe.mrp` in the original CoNLL-19 release.
    # Note that this steps removes sentences that are empty; we add them back in mrp_to_rdf.py
    python mtool/main.py --read conllu --write mrp --text ${DATASET_DIR}/${SPLIT}.lines ${DATASET_DIR}/${SPLIT}.conllu ${DATASET_DIR}/${SPLIT}.udpipe.mrp
    cd ${PROJECT_DIR}
    echo "Converting to MRP"
    # The resulting file is equivalent to the `evaluation/input.mrp` in the original CoNLL-19 release.
    python scripts/udpipe_to_mrp.py ${DATASET_DIR}/${SPLIT}.udpipe.mrp ${DATASET_DIR}/${SPLIT}.mrp
    cd ${PROJECT_DIR}/..
    echo "Preprocessing"
    mkdir ${DATASET_DIR}/${SPLIT}_aug_mrp
    python HIT-SCIR-CoNLL2019/toolkit/preprocess_eval.py ${DATASET_DIR}/${SPLIT}.udpipe.mrp ${DATASET_DIR}/${SPLIT}.mrp --outdir ${DATASET_DIR}/${SPLIT}_aug_mrp

    # We parse all formalisms in parallel. If we have >= 4 GPUs, parse on separate GPUs.
    cd ${PROJECT_DIR}/../HIT-SCIR-CoNLL2019
    GPU_IDX=0
    for FORMALISM in dm psd eds ucca; do
        case $FORMALISM in

        dm | psd)
            predictor_class="transition_predictor_sdp"
            ;;

        eds)
            predictor_class="transition_predictor_eds"
            ;;

        ucca)
            predictor_class="transition_predictor_ucca"
            ;;
        esac

        echo "Parsing ${FORMALISM}"
        allennlp predict \
            --cuda-device $GPU_IDX \
            --batch-size 32 \
            --output-file ${DATASET_DIR}/${SPLIT}.${FORMALISM}.mrp \
            --predictor ${predictor_class} \
            --include-package utils \
            --include-package modules \
            ../HIT-SCIR-CoNLL2019-model/${FORMALISM}/${FORMALISM}.tar.gz \
            ${DATASET_DIR}/${SPLIT}_aug_mrp/${FORMALISM}.mrp &

        if [[ $N_VISIBLE_GPUS -ge 4 ]]; then
            ((GPU_IDX=GPU_IDX+1))
        fi
    done

    wait

    cd ${PROJECT_DIR}
    echo "Converting to rdf"
    for FORMALISM in dm psd eds ucca; do
        python scripts/mrp_to_rdf.py ${DATASET_DIR}/${SPLIT}.${FORMALISM}.mrp ${DATASET_DIR}/${SPLIT}.${FORMALISM}.rdf ${DATASET_DIR}/${SPLIT}.${FORMALISM}.metadata
    done
done

echo "All done!"
