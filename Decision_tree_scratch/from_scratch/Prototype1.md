# Prototype 1 :

## Status : Failed  ⚠️

The attempt to make a flat , rule - induction approach using sequential Gini scanning was not of a great success.

**Things that worked :** 

    -Gini Impurity measurement worked flawless.
    -Calculation of each Gini Impurity checked for each feature went well with the Min Gini of each feature was used for the branching condition/threshold.
    -Extraction of Rules(The threshold that lead to a leaf) was extracted well.
    -Thresholds along with the feature index were packaged well and extracted well in prediction.

**Where things messed up:**

    -Though the aquired threshold map the prection failed to produce any result due to all the threshold map that created a leaf would lead to a empty array due to the flat , simple architecture and eastablishment of correlation between the data.

## Plan:

Current plan revolves around tackling the lack of branches and probabilities with Node based branching system.