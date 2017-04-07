++++++++++++++++++++++++++++++++++
# Switch GKE project
#+++++++++++++++++++++++++++++++++++++++++++++

gcloud config set project juju-davidson

#get active project
PROJECT_ID = $(gcloud config list project --format 'value(core.project)')

#defining bucket name
BUCKET_NAME = ${PROJECT_ID}-ml

#creating a bucket
gsutil mb -l europe-west1 gs://$BUCKET_NAME


#+++++++++++++++++++++++++++++++++++++++++++++
# Run on Cloud ML
#+++++++++++++++++++++++++++++++++++++++++++++


JOB_NAME = "TFHelloWorld"

STAGING_BUCKET = gs://$BUCKET_NAME

#running

gcloud ml-engine jobs submit training ${ JOB_NAME} \
--package-path = TFHelloWorld \
--staging-bucket = ${ STAGING_BUCKET } \
--module-name = TFHelloWorld.main


#+++++++++++++++++++++++++++++++++++++++++++++


gsutil mb -l europe-west1 gs://juju-davidson-ml


gcloud ml-engine jobs submit training TFShallowMNIST3 \
--package-path=TFHelloWorld \
--staging-bucket=gs://juju-davidson-ml/TFShallowMNIST3 \
--module-name=train.main \
-- --input_dir=input \
-- --output_dir=output


#+++++++++++++++++++++++++++++++++++++++++++++
# create dir on GS

gsutil cp /dev/null gs://juju-davidson-ml/TFShallowMNIST/

gsutil cp /dev/null gs://juju-davidson-ml/TFShallowMNIST3/input/

gsutil cp /dev/null gs://juju-davidson-ml/TFShallowMNIST3/output/

gsutil cp /dev/null gs://juju-davidson-ml/TFShallowMNIST3/output/model/

gsutil cp /dev/null gs://juju-davidson-ml/TFShallowMNIST3/output/summary_train/

gsutil cp /dev/null gs://juju-davidson-ml/TFShallowMNIST3/output/summary_test/

gsutil cp /dev/null gs://juju-davidson-ml/TFShallowMNIST3/output/model/
