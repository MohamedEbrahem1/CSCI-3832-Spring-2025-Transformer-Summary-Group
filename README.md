# CSCI-3832-Spring-2025-Transformer-Summary-Group
Project for CSCI 3832 Spring 2025 involving making a transformer model to perform summarization based on the overall reviews of food items from Amazon.

Our 3 main models are the RNN, BART, and T5 models.

Miles Zheng: To run the BART model download the dataset from https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews and move the kagglehub/datasets/snap/amazon-fine-food-reviews/versions/2/Reviews.csv into the same directory as the notebook. Then just run all cells, the model is trained with one cpu so it might take a while, the expected output is 

Initial Model Metrics:
ROUGE-L (Initial pre-trained model): 0.0711
Fine-tuned Model Metrics:
ROUGE-L (Fine-tuned model): 0.1335

and there are example reviews randomly printed out in the final cell.
Conda env: Python=3.10, datasets, evaluate, transformers=4.50, kagglehub=0.3.8, torch=2.6.0, rouge-score

Tian Zhang: LSTM
