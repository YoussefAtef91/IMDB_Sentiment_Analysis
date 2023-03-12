# IMDB_Sentiment_Analysis
# Context
Building a sentiment analysis model using IMDB reviews dataset


# Dataset
In this proejct, we're gonna use the dataset from [the Stanford page of Andrew Maas](https:/ /ai.stanford.edu/~amaas/data/sentiment)

# Data directory structure
aclImdb
├── test
│   ├── neg
│   └── pos
└── train
    ├── neg
    ├── pos
    └── unsup

7 directories

# Baseline Model and Performance Measure
Since the dataset is a balanced two-class classification, our naive baseline would be 50%, and the performance measure would be the accuracy


# Models Arcitectures
1.  Binary Unigram Dense Model
  *   TextVectorization => output_mode(multi_hot) + ngrams(1)
  *   Input_layer => shape = (20000,)
  *   Hidden_layers => Dense(16,relu) + Dropout(0.5)
  *   Output_layer => Dense(1, sigmoid)
  *   Accuracy: 88.6%
2.   Binary Bigram Dense Model
  *   TextVectorization => output_mode(multi_hot) + ngrams(2)
  *   Input_layer => shape = (20000,)
  *   Hidden_layers => Dense(16,relu) + Dropout(0.5)
  *   Output_layer => Dense(1, sigmoid)
  *   Accuracy: 89.3%
3.   Binary TF-IDF Dense Model
  *   TextVectorization => output_mode(tf_idf) + ngrams(2)
  *   Input_layer => shape = (20000,)
  *   Hidden_layers => Dense(16,relu) + Dropout(0.5)
  *   Output_layer => Dense(1, sigmoid)
  *   Accuracy: 87.7%
4.   One-Hot Bidirectional LSTM Model
  *   TextVectorization => output_mode(int)
  *   Input_layer => shape = (20000,)
  *   Hidden_layers => one_hot() + Bidirectioanl_LSTM(32) + Dropout(0.5)
  *   Output_layer => Dense(1, sigmoid)
  *   Accuracy: 87.9%
5.   Embedding Bidirectional LSTM Model
  *   TextVectorization => output_mode(int)
  *   Input_layer => shape = (20000,)
  *   Hidden_layers => Embedding(256) + Bidirectioanl_LSTM(32) + Dropout(0.5)
  *   Output_layer => Dense(1, sigmoid)
  *   Accuracy: 86.2%
6.   Pre-trained Embedding Bidirectional LSTM Model
  *   TextVectorization => output_mode(int)
  *   Input_layer => shape = (20000,)
  *   Hidden_layers => Embedding(100) + Bidirectioanl_LSTM(32) + Dropout(0.5)
  *   Output_layer => Dense(1, sigmoid)
  *   Accuracy: 87.6%
7. TransformerEncoder Model
  *   TextVectorization => output_mode(int)
  *   Input_layer => shape = (20000,)
  *   Hidden_layers => Embedding(256) + TransformerEncoder(32) + GlobalMaxPooling1D() + Dropout(0.5)
  *   Output_layer => Dense(1, sigmoid)
  *   Accuracy: 87.3%
8. TransformerEncoder with PositionalEmbedding Model
  *   TextVectorization => output_mode(int)
  *   Input_layer => shape = (20000,)
  *   Hidden_layers => PositionalEmbedding(256) + TransformerEncoder(32) + GlobalMaxPooling1D() + Dropout(0.5)
  *   Output_layer => Dense(1, sigmoid)
  *   Accuracy: 88.2%

# Conclusion
The Binary Bigram Dense model outperforms the other models with %89.3 accuracy
