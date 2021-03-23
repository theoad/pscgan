#!/bin/bash
methods="pscgan ours_mse lag dncnn"
datasets="ffhq church bedroom"
if [ -z ${n_gpus} ]; then
    echo "n_gpus environment variable empty. Setting n_gpus=1."
    export n_gpus=1
fi
if [ -z ${divide_expanded_forward_pass} ]; then
    echo "divide_expanded_forward_pass environment variable empty. Setting divide_expanded_forward_pass=1."
    export divide_expanded_forward_pass=1
fi

if [ -z ${thumbnails128x128} ]; then
    if [ ! -d ffhq_preprocessed ]; then
      echo "thumbnails128x128 environment variable empty and ffhq_preprocessed directory not found. Aborting."
      exit
    else
      echo "thumbnails128x128 environment variable empty. Skipping FFHQ preprocessing."
    fi
else
    python preprocess.py --dataset ffhq --data_dir ${thumbnails128x128} --out_dir ffhq_preprocessed \
    | tee reproduced_all/preprocessing_ffhq.txt
    echo "Done preprocessing ffhq"
fi

if [ -z ${church_outdoor_train_lmdb} ]; then
    if [ ! -d church_preprocessed ]; then
      echo "church_outdoor_train_lmdb environment variable empty and church_preprocessed directory not found. Aborting."
      exit
    else
      echo "church_outdoor_train_lmdb environment variable empty. Skipping FFHQ preprocessing."
    fi
else
    python preprocess.py --dataset church --data_dir ${church_outdoor_train_lmdb}  --out_dir church_preprocessed \
    | tee reproduced_all/preprocessing_church.txt
    echo "Done preprocessing Church Outdoor"
fi

if [ -z ${bedroom_train_lmdb} ]; then
    if [ ! -d bedroom_preprocessed ]; then
      echo "bedroom_train_lmdb environment variable empty and bedroom_preprocessed directory not found. Aborting."
      exit
    else
      echo "bedroom_train_lmdb environment variable empty. Skipping FFHQ preprocessing."
    fi
else
    python preprocess.py --dataset bedroom --data_dir ${bedroom_train_lmdb}  --out_dir bedroom_preprocessed \
    | tee reproduced_all/preprocessing_bedroom.txt
    echo "Done preprocessing Bedroom"
fi

rm -rf reproduced_all
mkdir reproduced_all

Downloading checkpoints
python download_checkpoint.py --all --out_dir checkpoints | tee reproduced_all/downloading_checkpoints.txt

# Figure 1 and 2
mkdir -p reproduced_all/Figures1and2
for METHOD in $methods; do
  for NOISE_STD in 25 50 75; do
    python test.py \
    --n_gpus ${n_gpus} \
    --method ${METHOD} \
    --noise_std ${NOISE_STD} \
    --save_batch 0 3 4 5 14 15 23 25 26 \
    --checkpoint checkpoints/ffhq/noise${NOISE_STD}/${METHOD}-${NOISE_STD}.ckpt \
    --test_set ffhq_preprocessed/test \
    --divide_expanded_forward_pass ${divide_expanded_forward_pass} \
    --out_dir reproduced_all/Figures1and2/${METHOD}_${NOISE_STD} \
    | tee reproduced_all/Figures1and2/${METHOD}_${NOISE_STD}.txt
  done
done
echo "Done Figure 1 and 2"

# Table 1
mkdir -p reproduced_all/Table1
for DATASET in $datasets; do
  for NOISE_STD in 25 50 75; do
    if [ "${DATASET}" == "ffhq" ]; then
      SIGMA_Z=1
    else
      SIGMA_Z=0.75
    fi

    python test.py \
    --n_gpus ${n_gpus} \
    --method pscgan \
    --noise_std ${NOISE_STD} \
    --fid_and_psnr \
    --num_fid_evals 32 \
    --sigma_z ${SIGMA_Z} \
    -N 64 \
    --checkpoint checkpoints/${DATASET}/noise${NOISE_STD}/pscgan-${NOISE_STD}.ckpt \
    --train_set ${DATASET}_preprocessed/train \
    --test_set ${DATASET}_preprocessed/test \
    --divide_expanded_forward_pass ${divide_expanded_forward_pass} \
    --out_dir reproduced_all/Table1/pscgan_${DATASET}_${NOISE_STD} \
    | tee reproduced_all/Table1/pscgan_${DATASET}_${NOISE_STD}.txt
  done
done

for METHOD in ours_mse dncnn; do
  for DATASET in $datasets; do
    for NOISE_STD in 25 50 75; do
      python test.py \
      --n_gpus ${n_gpus} \
      --method ${METHOD} \
      --noise_std ${NOISE_STD} \
      --fid_and_psnr \
      --checkpoint checkpoints/${DATASET}/noise${NOISE_STD}/${METHOD}-${NOISE_STD}.ckpt \
      --train_set ${DATASET}_preprocessed/train \
      --test_set ${DATASET}_preprocessed/test \
      --divide_expanded_forward_pass ${divide_expanded_forward_pass} \
      --out_dir reproduced_all/Table1/${METHOD}_${DATASET}_${NOISE_STD} \
      | tee reproduced_all/Table1/${METHOD}_${DATASET}_${NOISE_STD}.txt
    done
  done
done
echo "Done Table 1"

# Figure 3
mkdir -p reproduced_all/Figure3
for METHOD in pscgan lag; do
  for NOISE_STD in 25 50 75; do
    python test.py \
    --n_gpus ${n_gpus} \
    --method ${METHOD} \
    --noise_std ${NOISE_STD} \
    --fid_and_psnr \
    --num_fid_evals 32 \
    --sigma_z 0 0.25 0.5 0.75 1 \
    -N 1 2 4 8 16 32 64 \
    --checkpoint checkpoints/ffhq/noise${NOISE_STD}/${METHOD}-${NOISE_STD}.ckpt \
    --train_set ffhq_preprocessed/train \
    --test_set ffhq_preprocessed/test \
    --divide_expanded_forward_pass ${divide_expanded_forward_pass} \
    --out_dir reproduced_all/Figure3/${METHOD}_${NOISE_STD} \
    | tee reproduced_all/Figure3/${METHOD}_${NOISE_STD}.txt
  done
done
echo "Done Figure 3"

# Figure 4
mkdir -p reproduced_all/Figure4
for METHOD in pscgan ours_mse; do
  for NOISE_STD in 25 50 75; do
    python test.py \
    --n_gpus ${n_gpus} \
    --method ${METHOD} \
    --noise_std ${NOISE_STD} \
    --denoiser_criteria \
    --checkpoint checkpoints/ffhq/noise${NOISE_STD}/${METHOD}-${NOISE_STD}.ckpt \
    --test_set ffhq_preprocessed/test \
    --divide_expanded_forward_pass ${divide_expanded_forward_pass} \
    --out_dir reproduced_all/Figure4/${METHOD}_${NOISE_STD} \
    | tee reproduced_all/Figure4/${METHOD}_${NOISE_STD}.txt
  done
done
echo "Done Figure 4"


# Figure 6 and 7
mkdir -p reproduced_all/Figures6and7
for DATA_SET in bedroom church; do
  for METHOD in pscgan ours_mse dncnn; do
    for NOISE_STD in 25 50 75; do
      python test.py \
      --n_gpus ${n_gpus} \
      --method ${METHOD} \
      --noise_std ${NOISE_STD} \
      --save_batch 0 1 3 4 6 8 9 \
      --checkpoint checkpoints/${DATA_SET}/noise${NOISE_STD}/${METHOD}-${NOISE_STD}.ckpt \
      --test_set ${DATA_SET}_preprocessed/test \
      --divide_expanded_forward_pass ${divide_expanded_forward_pass} \
      --out_dir reproduced_all/Figures6and7/${METHOD}_${NOISE_STD} \
      | tee reproduced_all/Figures6and7/${METHOD}_${NOISE_STD}.txt
    done
  done
done
echo "Done Figure 6 and 7"