#!/usr/bin/env bash

#-----------------------------------
#----- Training on Cloud ML
#-----------------------------------

/anaconda/bin/python -m train.model \
--output_dir ./output \
--number 122323232434344323

#-----------------------------------
#----- Training on Cloud ML
#-----------------------------------

gcloud ml-engine jobs submit training TF_Babylonian_job_0 \
--package-path train \
--job-dir gs://dubai-tf-meetup-v2/TF_Babylonian \
--module-name train.model \
--region europe-west1 \
-- \
--number 122323232434344323


