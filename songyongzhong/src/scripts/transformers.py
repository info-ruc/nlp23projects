# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("question-answering", model="jolual2747/qa_nlp_model")

main