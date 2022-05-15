# Comparative Analysis of Models for Sentiment Analysis

1.) This project aims to compare 5  models and test the reliability of hybrid sentiment analysis models on various datasets of different domains.<br>

2.) Determining whether it is possible to produce hybrid models that outperform single models with different domains and types of datasets.

## Datasets

Two datasets are used for comparison:<br>

1.) IMDb Movie review dataset: 50000 reviews<br>

2.) Amazon Product review dataset: ~100000 reviews<br>

These datasets are divided into two labels for sentiment: Positive and Negative.

## Models 

We have used five models for comparison:<br>

1.) LSTM<br>

2.) CNN<br>

3.) QRNN<br>

4.) CNN-LSTM<br>

5.) LSTM-CNN<br>

LSTM is a network model that solves the problems of gradient explosion and vanishing gradient in RNN. It is widely used in sentiment analysis and text analysis, as it has its own memory and therefore has the advantages of analyzing relationships among time series data through its memory function. It can also make relatively accurate forecasting.

LSTM has a memory cell at the top which helps to carry the information from a particular time instance to the next time instance in an efficient manner. Therefore it can able to remember a lot of information from previous states. This information can be added or removed from the memory cell using valves. In the LSTM network, the total input is the input data from the current time instance and the output of hidden layer from the previous time instance. These are then passed through activation functions and valves in the network and then output is produced.


CNN  can also be used for text analysis. CNN's can process local features such as 

n-grams in text. We can use CNNs for sentiment analysis by using the word to int vectors.

The word vectors act like Images as some points in space. By using word-to-Int vectors of fixed size and stacking them on top of each other, we get an “image”. Therefore we can use CNN on text similar to how we use image classification. For NLP, we typically use filters that slide over word embeddings — matrix rows.

Text data needs to be analysed in the same way as people read the text: word by word, from the beginning to the end, taking context into account as one go.  This conventional one-word-at-a-time approach is called a "recurrent neural network". Unlike the conventional RNN, in the "quasi-recurrent neural network," or QRNN most of the computation happens all at once parallelly which can make it up to 16 times faster than the old approach. QRNN approach can also help with the interpretability of neurons, because each neuron's activation doesn't depend on the past history of any other neurons. The neurons thus can have (although not guaranteed) independent and well-defined meanings and these meanings are more likely to be simpler and more human-interpretable.

Using Combinations of CNN and LSTM can mean that we used the best of both worlds.

CNN is able to learn the features that contribute significantly to the sentiment of the sentence using overlapping sequences resulting in a matrix of word vectors. This matrix can then be fed to LSTM. The LSTM layer can then focus on more important data rather than memorizing the whole sentence. This can lead to an increase in the efficiency of the model. 



## Preprocessing

The first step is to pre-process the data. The preprocessing steps include:<br>

1.) Removing stop words, punctuations, html parsers etc.<br>

2.) Converting to lower case.<br>

3.) Stemming/Lemmatization.<br>

The files for preprocessing in the repo are:

1.)amazon\_reviews\_preprocess.ipynb

2.)imdb\_preprocess.ipynb

These files will:

1.) create a file preprocessed\_data.pkl (IMDb dataset) and preprocessed\_data\_ammazon.pkl (Amazon dataset) after text cleaning in cache/sentiment\_analysis/ directory.

2.) Create dictionary for both the datasets(word\_dict\_qrnn.pkl for IMDb and word\_dict\_amazon.pkl for Amazon dataset).

3.) Perform word to int mapping for both the datasets:

`	`IMDb: train\_qrnn.csv, test\_qrnn.csv

`	`Amazon: train\_amazon.csv, test\_amazon.csv

4.) Dictionary and Word to Int Mappings can be found in /data/pytorch directory

After these files are created we can proceed to training.

## Model Training and Testing 

### Training files for all models in repo

(First run the preprocessing step, then run the files from the marked cell)

### Experimental Setup:

1) Used HPC Cluster. GPU: RTX8000
1) Train and test data is mixed and separated to 80%(train), 10%(val) and 10%(test)
1) Ran the training for 10 Epochs for all runs (to maintain uniformity)

First we trained and tested the models on both the datasets using the 5 models specified above with some default parameter.

**Model parameters for IMDb and Amazon Dataset:**

||LSTM|QRNN|CNN|CNN-LSTM|LSTM-CNN|
| :- | :-: | :-: | :-: | :-: | :-: |
|Optimizer|Adam|Adam|Adam|Adam|Adam|
|LR|0.001|0.001|0.001|0.001|0.001|
|Dropout|0|0.3|0.5|0.5|0.5|
|Hidden size/Num Layers(qrnn)|100|2|-|100|100|
|Embed\_size|32|32|32|32|32|
|Filter size/num\_filters(qrnn)|-|64|100|100|100|



We then used Hyperband optimizations to get the optimum values of parameters. We tested the optimized models and compared the results.

**Model Parameters for IMDb Dataset**

||LSTM|QRNN|CNN|CNN-LSTM|LSTM-CNN|
| :- | :- | :- | :- | :- | :- |
|Optimizer|RMSProp|RMSprop|Adam|Adam|RMSprop|
|LR|0.0038|0.0003|0.0004|0.0091|0.145|
|Dropout|0.3|0.189|0.3764|0.367|0.381|
|Hidden size/Num Layers(qrnn)|215|3|-|117|56|
|Embed\_size|112|93|54|31|119|
|Filter size/num\_filters(qrnn)|-|57|115|103|172|


**Model Parameters for Amazon Dataset**

||LSTM|QRNN|CNN|CNN-LSTM|LSTM-CNN|
| :- | :- | :- | :- | :- | :- |
|Optimizer|Adam|Adam|RMSprop|RMSprop|Adam|
|LR|0.8|0.0001|0.00301|4.69|0.354|
|Dropout|0.14|0.24|0.5373|0.535|0.148|
|Hidden size/Num Layers(qrnn)|247|4||159|188|
|Embed\_size|55|52|73|45|40|
|Filter size/num\_filters(qrnn)|-|176|56|93|92|


\## Results

Results are plotted in the file:

1) [Amazon_graphs-Hyperband.ipynb](https://github.com/Shr2020/IDLSProject/blob/main/Amazon_graphs-Hyperband.ipynb)
1) [Amazon_graphs.ipynb](https://github.com/Shr2020/IDLSProject/blob/main/Amazon_graphs.ipynb)
1) [IMDB_graphs-FINAL.ipynb](https://github.com/Shr2020/IDLSProject/blob/main/IMDB_graphs-FINAL.ipynb)
1) [IMDB_graphs-Hyperband.ipynb](https://github.com/Shr2020/IDLSProject/blob/main/IMDB_graphs-Hyperband.ipynb)
1) [Hyperband vs Non-Hyperband AMAZON.ipynb](https://github.com/Shr2020/IDLSProject/blob/main/Hyperband%20vs%20Non-Hyperband%20AMAZON.ipynb)
1) [Hyperband vs Non-Hyperband IMDB.ip](https://github.com/Shr2020/IDLSProject/blob/main/Hyperband%20vs%20Non-Hyperband%20IMDB.ipynb)[ynb](https://github.com/Shr2020/IDLSProject/blob/main/Hyperband%20vs%20Non-Hyperband%20IMDB.ipynb)


### IMDB Dataset:




### Amazon Dataset







### IMDb Dataset: With Hyperband Optimization





### Amazon Dataset: With Hyperband Optimization





### Comparing the performance of Two approaches (Hyperband and Non-Hyperband)

### IMDb Dataset




### Amazon dataset





### Observations:

- The Accuracy for all models is the same more or less. CNN-LSTM  performed the best looking at the graphs for the IMDb dataset. For the Amazon Dataset, LSTM performed the best.
- LSTM-CNN suffers from overfitting. Hence its Accuracy is the least for both the Datasets.
- QRNN takes less time than half time than LSTM as expected.
- CNN and CNN-LSTM were the most time consuming when run without Hyperband optimisation.
- The Accuracy for all the models increases or remains the same after Hyperband Optimization.
- The train time for CNN and CNN-LSTM decreases with Hyperband Optimization. This may be because the hyperband chooses the optimisers and the right optimisers can reduce the training time significantly.

