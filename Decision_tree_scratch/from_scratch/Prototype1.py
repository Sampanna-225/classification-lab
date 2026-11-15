import numpy as np
import pandas as pd

train1 = np.array([
    [0.2, 0.1],  
    [0.4, 0.3],  
    [0.3, 0.2],  
    [0.5, 1.0],  
    [1.0, 0.2],  
    [1.2, 0.1],  
    [1.1, 0.3],  
    [1.6, 1.1], 
    [1.8, 1.3],  
    [1.5, 1.2],  
], dtype=np.float32)

train2 = np.array([
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
])

#rule based
class DecisioTree:
    def __init__(self , X:np.array , Y:np.array):
        self.x = X
        self.y = Y
        self.data_predict = None
        
    def Process(self):
        X_left , X_temp = self.x , self.x
        Y_left , Y_temp = self.y , self.y
        
        result = []
        temp =[]
        threshold_list = (
            []
        )
        # X_right , Y_right = np.zeros(X_left.shape)  , np.zeros(Y_left.shape)
        
        while True:
            if len(X_left) == 0:
                break
            features = [X_left[:,i] for i in range(X_left.shape[1])]
            best_gini = float('inf')
            best_index = None
            best_threshold = None
            last_leaf = None


            for  n,chunk in enumerate(features): #enumerate gives both index and the value
                GiniList = (
                    []
                )
                for i in chunk: 
                    mask = X_left[:,n] <= i
                    TempArray = Y_left[mask]
                    GiniList.append(self.Gini(TempArray))

                if min(GiniList) < best_gini: # always need to reduce Gini each turn
                    best_gini = min(GiniList)
                    best_index = n
                    best_threshold = chunk[np.argmin(GiniList)]
                    mask = X_left[:,n] <= best_threshold
                    X_left , Y_left  =X_left[mask] , Y_left[mask]
                    threshold_list.append((best_threshold , best_index))
                    last_leaf = np.unique(Y_left , axis = 0)
                    

                elif best_index == None and best_threshold == None :
                    break

                
            result.append((list(threshold_list), last_leaf ))
            filter = np.any(Y_temp != last_leaf , axis=1)
            X_left , Y_left = X_temp[filter] , Y_temp[filter]
            X_temp , Y_temp = X_temp[filter] , Y_temp[filter]
            threshold_list.clear()

        self.data_predict = result
        

    def predict(self, x:np.array):
        rules_col = []
        class_col = []
        status_col = []

        for sample in self.data_predict:
            rules = sample[0]
            predicted_class = sample[1]
            
            rules_col.append(rules)
            class_col.append(predicted_class)

            guessX = x.copy()
            for tup in rules:
                mask = guessX[:, tup[1]] <= tup[0]
                guessX = guessX[mask]
                
            if len(guessX) == 0:
                status_col.append("Threshold didnt match")
            else:
                status_col.append("It matched")

       
        df = pd.DataFrame({
            "Rules/Thresholds": rules_col,
            "Predicted Class": class_col,
            "Status": status_col
        })
        print(df)
        
            
    def Gini(self , labA:np.array):
        count_list = []
        gini_list = []
        while True:
            if len(labA) == 0:
                return 1.0
            a = labA[0]
            mask = np.all(labA == a, axis=1)#acts as AND gate every element must be true
            temp = labA[mask]
            count_list.append(len(temp))
            mask = np.any(labA != a , axis=1)#acts as OR gate even if single feature is true it allows it which means if even a feature is not as in index[0] it returns true
            labA = labA[mask]


            if len(labA)==0:
                break

        total = sum(count_list)
        sumP = 0
        for i in count_list:
            P = (i/total)
            sumP += P**2

        gini_impurity = 1 -sumP
        return gini_impurity



tempt = DecisioTree(train1, train2)
tempt.Process()
tempt.predict(np.array([[1.6,1.2]]))

