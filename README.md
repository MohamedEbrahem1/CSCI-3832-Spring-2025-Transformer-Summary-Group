# CSCI-3832-Spring-2025-Transformer-Summary-Group
Project for CSCI 3832 Spring 2025 involving making a transformer model to perform summarization based on the overall reviews of food items from Amazon.

Our 3 main models are the RNN, BART, and T5 models.

Ryen Johnston: To prepare the data, first download Reviews.csv from https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews and place it in the root directory. Then, run preprocesser.py. This will output a new csv called "FilteredReviews.csv" in the root directory.
This file can be used for all models.

Miles Zheng: To run the BART model use the bart_summarizer notebook, can download the dataset as a zip from https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews or in the notebook there is some code that downloads the dataset. Then the kagglehub/datasets/snap/amazon-fine-food-reviews/versions/2/Reviews.csv into the same directory as the notebook. Then just run all cells, the model is trained with one cpu so it might take a while, the expected output is 

Initial Model Metrics:
ROUGE-L (Initial pre-trained model): 0.0711
Fine-tuned Model Metrics:
ROUGE-L (Fine-tuned model): 0.1335

and there are example reviews randomly printed out in the final cell.
Conda env: Python=3.10, datasets, evaluate, transformers=4.50, kagglehub=0.3.8, torch=2.6.0, rouge-score

Tian Zhang: LSTM run the ForwardRNN.py. file pass in the Reviews.csv file or FilteredReviews.csv file into the process_reviews function. i.e. in main, write model = train_model('Reviews.csv')
This trains the model on the reviews dataset. This and the following step are all done in if __name__ == '__main__':

To specify what reviews you'd like to summarize,  call process reviews in this fashion in main: process_reviews('item1.csv', model, word2id) with model being the model that was created
by train_model("csv file") and the csv file that contains reviews that you'd like to summarize. There are already 3 item csvs files preincluded that was a test set that we used to test
our trained models. 

Mohamed Abdelmagid: T5 model run the t5_pretrained.ipynb file inside the archive folder. Create a models and data folder in the parent directory (../) and put the filtered_reviews.csv (or rename reviews.csv to that) in the data folder. Then run and train the model and you should get

Initial Model Metrics:
ROUGE-L (Initial pre-trained model): 0.0890
BLEU (Initial pre-trained model): 0.0036
METEOR (Initial pre-trained model): 0.1272

Fine-tuned Model Metrics:
ROUGE-L (Fine-tuned model): 0.1432
BLEU (Fine-tuned model): 0.0000
METEOR (Fine-tuned model): 0.0865

and the final cell prints out examples.
