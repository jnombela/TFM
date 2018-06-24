#!/usr/bin/env python3
# -*- coding: utf-8 -*-

export JOB_NAME=job_4_4
export MODULE=trainer.4_train
export PACKAGE_PATH=./trainer
export BUCKET=gs://deep-1-203210-mlengine
export RUTA_TRAIN=/mini-data/train
export RUTA_VALID=/mini-data/validation
export RUTA_CONFIG=trainer/cloudml-gpu.yaml
export REGION=europe-west1
export RUNTIME_VERSION=1.8

gcloud ml-engine jobs submit training ${JOB_NAME} \
    --module-name="${MODULE}" \
    --staging-bucket="${BUCKET}" \
    --package-path="${PACKAGE_PATH}" \
    --config "${RUTA_CONFIG}" \
    --region "${REGION}" \
    --runtime-version "${RUNTIME_VERSION}" \
    -- \
    --rtrain "${BUCKET}${RUTA_TRAIN}" \
    --rvalid "${BUCKET}${RUTA_VALID}"
    
    
