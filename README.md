# wiki2bio
This project provides the implementation of table-to-text (infobox-to-biography) generation, taking the structure of a infobox for consideration.

Details of table-to-text generation can be found [here](https://tyliupku.github.io/papers/aaai2018_liu.pdf). The implementation is based on Tensorflow 1.0.0 and Python 2.7.

# Model Overview
<p align="center"><img width="85%" src="doc/frame.png"/></p>
wiki2bio is a natural language generation task which transforms Wikipedia infoboxes to corresponding biographies.
We encode the structure of an infobox by taking field type and position information into consideration.

In the encoding phase, we update the cell memory of the LSTM unit by a field gate and its corresponding field value 
in order to incorporate field information into table representation.
In the decoding phase, dual attention mechanism which contains word level attention and field level attention is proposed 
to model the semantic relevance between the generated description and the table.

# Data
The dataset for evaluation is [WIKIBIO](https://github.com/DavidGrangier/wikipedia-biography-dataset) from [Lebret et al. 2016](https://arxiv.org/abs/1603.07771). We preprocess the dataset to a easy-to-use way.

```*.box``` in the ```original_data``` is the infoboxes from Wikipedia. One infobox per line.

```*.summary``` in the ```original_data``` is the biographies corresponding to the infoboxes in ```*.box```. One biography per line.

The whole dataset is divided into training set (582,659 instances, 80%), valid set (72,831 instances, 10%) and testing set (72,831 instances, 10%).

# Usage
To be done...
