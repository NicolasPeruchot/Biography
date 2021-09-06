# Automatic Biography Creator 

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


<p align="center">
<img src="https://github.com/NicolasPeruchot/Biography/blob/main/Example.png" alt="drawing" width="500"/>
</p>

## About 

A.B.C is an NLP project realised during my master's degree at [CentraleSupélec.](https://www.centralesupelec.fr/en/school-0)

The purpose of this project is to generate short biographies with few informations found on WikiData, using up-to-date models.

The work in this repo is divided in three part:

- Cleaning the dataset, using spaCy
- Training the model, using Huggingface
- Deploying the application, using Streamlit

## Usage

The model has been uploaded [as a HuggingFace repository](https://huggingface.co/NicolasPeruchot/Biography) and the application is implemented with [Streamlit](https://share.streamlit.io/nicolasperuchot/biography/main/steamlit/stream.py).


## Development

### Installation
``
make install
``

### Dataset generation

```make dataset```

### Model training

```make training```

### Launch application locally

``
make app
``


