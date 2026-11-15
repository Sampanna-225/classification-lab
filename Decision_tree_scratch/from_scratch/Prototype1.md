# Prototype 1 :

## Status : Success 🟢

The attempt to make a flat , rule - induction approach using sequential Gini scanning was not of a great success.

**Things that worked :** 

    -Gini Impurity measurement worked flawless.
    -Calculation of each Gini Impurity checked for each feature went well with the Min Gini of each feature was used for the branching condition/threshold.
    -Extraction of Rules(The threshold that lead to a leaf) was extracted well.
    -Thresholds along with the feature index were packaged well and extracted well in prediction.

**Where things messed up:**

### v1

    -Though the aquired threshold map the prection failed to produce any result due to all the threshold map that created a leaf would lead to a empty array due to the flat , simple architecture and eastablishment of correlation between the data.

    The faulty proces:
    for  n,chunk in enumerate(features): #enumerate gives both index and the value
                GiniList = (
                    []
                )
                for i in chunk: 
                    mask = X_left[:,n] <= i
                    TempArray = Y_left[mask]
                    GiniList.append(self.Gini(TempArray))
                

                if round(min(GiniList),2) == 0.0:
                    threshold = chunk[np.argmin(GiniList)]
                    mask = X_left[:,n] <= threshold
                    if n == 0 :
                        X_right , Y_right = X_left[~mask] , Y_left[~mask]
                    else:
                        pass
                    X_left , Y_left  =X_left[mask] , Y_left[mask]
                    threshold_list.append((threshold , n)) # to see which column the feature represents.
                    result.append((list(threshold_list), np.unique(Y_left , axis = 0)))
                    continue

                else:
                    threshold = chunk[np.argmin(GiniList)]
                    mask = X_left[:,n] <= threshold
                    if n == 0 :
                        X_right , Y_right = X_left[~mask] , Y_left[~mask]
                    else:
                        pass

                    X_left , Y_left  =  X_left[mask] , Y_left[mask]
                    
                    threshold_list.append((threshold , n)) # to see which column the feature represents.


            X_left , Y_left = X_right , Y_right

### v2

    Works flawlessly due to sequential rule based mapping. Instead of branching and branching it just separates the leaf we found from the original messed up pile to clear up and classify another using the same process filtering just like picking a colored bull of mixed up colored balls at a time till the pile is empty and everything is sorted.

## Plan:

Current plan revolves around tackling the lack of branches and probabilities with Node based branching system.