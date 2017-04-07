#!/usr/bin/env bash

gcloud ml-engine jobs submit training TFShallowMNIST_1 \
--package-path=train \
--job-dir=gs://dubai-tf-meetup/TF_SHALLOW_MNINST_1 \
--module-name=train.main \
--region=europe-west1 \
-- \
--output_dir=gs://dubai-tf-meetup-v2/TF_SHALLOW_MNINST_1/output \
--checkpoint_dir=gs://dubai-tf-meetup-v2/TF_SHALLOW_MNINST_1/output/model


