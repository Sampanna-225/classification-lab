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


 ## Decision Tree :

 A decision tree is a advanced flow chart algorithm which starts from a root parent node. Each next step asks a Yes-No question to the previsous node and filter in the one that satisfies the question. The satisfied new nodes are branched for further filtering or form a leaf (Purest form) the other separated data are fed into a separate branch to further ask more Y/N Questions to separate them even more. 

 Likely they start from a root node but branch and form a tree like shape closely resembling to a family tree. With series of Y/N qns the parent root is divided and classified into purest forms which works as the model's classification.

      Say we separate Mouse , Cat and Dog. We devide them by Height and Weight.

      Root Node0:
      If w<= 0.15 :
      - True : Mouse
      - False : Go to Node1

      Node1 :
      If w<= 4.50 :
      - True : Go to Node2
      - False : Dog

      Node2 :
      If h <= 3.5 (m) :
      - True : Cat
      - False : Dog

### Splitting / Purification testing

Key : A metric is a standardized, quantifiable measurement used to evaluate, track, or compare the performance, quality, or characteristics of something.

Gini Impurity is a metric used in decision tree to measure how often a randomly chosen element from a set would be incorrectly labeled if it were to be randomly be labeled according to the distribution of labels in subset.

Formulae:
$$Gini = 1 - \sum_{i=1}^{C} p_i^2$$

Where,

$p_i$ (Probability/Proportion): The proportion of samples that belong to class $i$ in that specific node 
$p_i = \frac{\text{Count of class } i}{\text{Total samples in node}}$

$C$ (Number of Classes): The total number of distinct target classes.

Range :
 0.0 : Purest form
 0.5 : Maximum impurity 50/50 odds

      An example :

      7 samples are Class A (e.g., Spam)

      3 samples are Class B (e.g., Not Spam)

#### Finding probability: 

$p_{\text{A}} = \frac{7}{10} = 0.7$

$p_{\text{B}} = \frac{3}{10} = 0.3$

#### Finding the squared sum :

$\sum p_i^2 = (0.7)^2 + (0.3)^2 = 0.49 + 0.09 = 0.58$

#### Finally Gini Impurity :

$Gini = 1 - 0.58 = \mathbf{0.42}$

50/50 is the absolute highest of the Gini score cause if say we have 7 cats and 3 dogs the machine still has a 70% chance of picking cat correctly while in 50/50 the machine is in a stalemate where there is equal chance of picking a cat while also equal chance of picking a dog at which point the machine is not classifing its guessing.

The most important fact we need to know is that Gini impurity is calculated at each step where we want to brach and ask Y/N qns until we hit closest to 0.0 (Absolute Purity)

### <mark> We dont write a single line of qns in Decision tree : </mark>
   The tree goes through every feature x and takes it as a threshold and makes a split. Right after it checks the Gini Impurity and when the gini impurity decreases it filters out the data making a split and impure data goes to a direction and pure data goes towards another both sides constantly branching till Gini Impurity 0.0 to form a leaf.

   Also not only the features X get masked and filtered their corresponding row of Output Y also accosiated also get masked too. Letting us know what catagory was fully purified.

      Numpy code :

      mask = X[:, 0] <= 2.10
      X_left, y_left, X_right, y_right = X[mask], y[mask], X[~mask], y[~mask]

   Here X[:, 0] slices the array including all rows of column index 0 (means taking only column 1)

   mask = X[:, 0] <= 2.10 all these return a boolean masking of either true or false like [True , True , False]

   ~ flips the boolean acting as a not gate turning True to False and vise-versa.

   Likely two branches are separated above with separated X and Y.