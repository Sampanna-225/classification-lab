# ALOGRITHMS 
<!-- (Ctrl + Shift + V) -->
## $k$-Nearest Neighbors ($k$-NN)

 $k$-NN is the simplest form of decision making there is. Instead of learning a complex equations or stratezizing insane weight and bias matrixes we simply plot the feature in a graph.

    Eg : Say we got a animal ear-size (x) and teeth-size (y) in graph.

 We simply plot the known data like data collected form cats , dogs and tigers.

 <mark> We assume that the data of each catagories are close to each other. </mark>

  Hense now when we add a new entry and let the Machine Learn the what it classifies at we take <mark>.' k '.</mark> variable, which is the number of nearest neighbours closest in distance to the new entry. The final step is to count highest number of neighbors of the same catagories that are closest to the new entry.

    Eg : Say we have 3 Cats , 1 Dog and 1 Tiger closest neighbors to the new entry we classify it as a cat.

 <mark> The Key note here is that ' k ' must be odd number to avoid ties . </mark>