install:
	python -m pip install -r requirements.txt

dataset:
	python -m dataset_creation.dataset

train:
	python -m model_creation.training
	
app:
	streamlit run streamlit/stream.py
