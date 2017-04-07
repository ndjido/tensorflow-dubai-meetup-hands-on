#!/usr/bin/env bash

#-----------------------------------
#----- Training on local
#-----------------------------------

/anaconda/bin/python -m train.model \
--output_dir ./output


#-----------------------------------
#----- Training on Cloud ML
#-----------------------------------

gcloud ml-engine jobs submit training TF_LM_job_3 \
--package-path model \
--job-dir gs://dubai-tf-meetup-v2/TF_ML \
--module-name train.model \
--region europe-west1 \
-- \
--output_dir gs://dubai-tf-meetup-v2/TF_LM/output

#-----------------------------------
#----- Online Prediction
#-----------------------------------

gcloud ml-engine predict \
--model "tf_lm" \
--version "v1" \
--json-instances ./input/outOfSample.json

