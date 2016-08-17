# Summer Networks ('16)
Neural Network examples and experiments performed during the summer of 2016. The first 2 sections are illustrative of how neural networks work in general, while on the last 2 I try to create a RNN that trains on Bukowski poems to write poems that look like his. All using Theano and Numpy, and following the excellent [WildML's tutorials](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/).

* [**Numpy Network**](https://github.com/masta-g3/summer_networks/blob/master/numpy_networks.ipynb): Base numpy-based implementation of a neural network for 2D classification.

* [**Theano Network**](https://github.com/masta-g3/summer_networks/blob/master/theano_networks.ipynb): Same problem as above, but using a Theano implementation instead.

* [**Numpy Bukowski**](https://github.com/masta-g3/summer_networks/blob/master/bukowski_networks/numpy_bukowski.ipynb): Recurrent neural network (RNN) using Numpy. Here I start using a collection of Bukowski's poems as training data.

* [**Theano Bukowski**](https://github.com/masta-g3/summer_networks/blob/master/bukowski_networks/theano_bukowski.ipynb): Reimplementation of the Bukowski neural network, but now using Theano, as the base Numpy model takes significantly longer to run. Sample output poems are available here.
