# ALOGRITHMS 
<!-- (Ctrl + Shift + V) -->
## $k$-Nearest Neighbors ($k$-NN)

 $k$-NN is the simplest form of decision making there is. Instead of learning a complex equations or stratezizing insane weight and bias matrixes we simply plot the feature in a graph.

    Eg : Say we got a animal ear-size (x) and teeth-size (y) in graph.

 We simply plot the known data like data collected form cats , dogs and tigers.

 <mark> We assume that the data of each catagories are close to each other. </mark>

  Hense now when we add a new entry and let the Machine Learn the what it classifies at we take <mark>.' k '.</mark> variable, which is the number of nearest neighbours closest in distance to the new entry. The final step is to count highest number of neighbors of the same catagories that are closest to the new entry.

    Eg : Say we have 3 Cats , 1 Dog and 1 Tiger closest neighbors to the new entry we classify it as a cat.

 <mark> The Key note here is that ' k ' must be odd number to avoid ties for binary classification 0s or 1s but if we have 3 classes a tie can still occur. </mark>

 A true fix can be distance weight voting:

 $w_i = \frac{1}{d_i + \epsilon}$

      where $\epsilon$ is a tiny number like 1e-8 to prevent division by zero if distance is 0.

 Like this we can multiply the votes with the weights and the highest valued vote must be used.

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
 0.5 : Maximum impurity 50/50 odds for 2 data else it can rise up to $\frac{K-1}{K}$.

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

Also in a decision tree weighted gini is used. Means each threshold is used to split and the last weighted gini determines the best split case:
$$\text{Weighted Gini} = \left(\frac{N_{\text{left}}}{N_{\text{total}}}\right)\text{Gini}(\text{left}) + \left(\frac{N_{\text{right}}}{N_{\text{total}}}\right)\text{Gini}(\text{right})$$



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

## Random Forest:

Random Forest is a combination of multiple decision tree which are fed datas collected of the same format and class . After training during testing phase each decision tree vote on a class. The class with the most amount of votes win.(We take average if the output is continuous unlike classification "0" or "1")


### Bootstrapping:

Instead of giving each tree in Random Forest whole dataset, we create a new training dataset of the same size and randomly pick the data allowing the replacement(repeatation) .

Eg: Lets say we have a dataset of size 5. So their row indexes are [0,1,2,3,4]

So randomly selected with replacements looks like [1, 3, 1, 4, 1]. You can see that 1 is repeated 3 times that means the data of index 1 is used 3 times in the decision tree.

This is done because a row can be picked multiple times, roughly 63.2% of your unique training rows end up inside a given tree's sample, while the remaining ~36.8% are left out. Each tree sees a slightly different slice of reality, so they don't all memorize the exact same noise.

Each decision tree gets its own separate bootstrapped sample, drawn fresh each time. This allows each decision tree to access its own unique training sets without compramizing the nature of the original dataset.

### Out-of-Bag (OOB) Samples:

That left over ~36.8% of rows that wasnt picked for the Tree 1 Bootstrapped dataset is called Tree 1's OOB data.

This is helpful for validation cause the left out data can be used as testset for the same decision tree because the tree never trained on those OOB rows, making them an unbiased, free validation set from the same distribution.

### Bagging:

We train "N" Trees on "N" Bootstrapped Dataset so each tree gets its own Bootstrapped dataset.

For classification we take majority vote. But for regression (continuous data prediction like [10,20,40]) we take the average value cause majority vote can make some predicitons left out.

In ML variance tells how much the prediction of the model changes based on the change in some elements of dataset. This means bootstapping can create much noticable changes if trained on small data or depening on the senistivity of the model.

Averaging/voting across many slightly different trees cancels out individual tree mistakes.

### Feature Sampling:

There is a huge problem with Bagging alone. If a feature is super-strong every tree will take it as the threshold feature meaning each tree may look identical.

We can fix it by only letting each tree access only int(np.sqrt(n)) where n is the number of feature the dataset has and the selected features are randomized.

Eg: a dataset has features [height , length , breadth , speed]

Index wise we write [0,1,2,3]


So Using feature sampling we take only 2 feature for dataset and randomly selected index list may be [1,3].

This is done at each node split so the next node has new randamized features to work through.

So the specific dataset only gets [height , speed] as features to use and for double combo randomness.


### Boosting:

Training trees sequentially where each new tree fixes the previous tree's errors (e.g., XGBoost), unlike Random Forest where trees train independently in parallel.

It can be used instead of bagging.