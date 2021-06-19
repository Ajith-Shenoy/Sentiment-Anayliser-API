# Sentiment-Anayliser-API
Restapi using fastapi for deploying multi-class sentiment analysis using 1DCNN Model 

trained on english reviews scraped from trustpilot.com , and based on ratings assigned thrree classes namely:
very positive , average , and poor.

Pytorch is used to train the model. refer to model.py for the architecture. And keras tokenizer is used for text tokenization . Glove 8B 300d fixed embeddings are used .
