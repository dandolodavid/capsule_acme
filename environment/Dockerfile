# hash:sha256:0430f8e73624ca307d654205b867d52f4b8011ec93d049b3e6e0baa7944bc012
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install -U --no-cache-dir \
    catboost==1.1 \
    kaleido==0.2.1 \
    matplotlib==3.6.1 \
    numpy==1.23.3 \
    pandas==1.5.0 \
    plotly==5.10.0 \
    scikit-learn==1.1.2 \
    shap==0.41.0 \
    statwolfacme==1.1.0 \
    xgboost==1.6.2
