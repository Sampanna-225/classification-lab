import numpy as np

class Decision_Tree:
    def __init__(self , X , Y):
        self.root_x = X
        self.root_y = Y
        

    class Node:
        def __init__(self ,outer_instance , X, Y , prev_value=None):
            self.x = X
            self.y = Y
            self.root = outer_instance
            self.right = None
            self.left = None
            self.x_left = None
            self.y_left = None
            self.x_right = None
            self.y_right = None
            self.best_gini = float('inf')
            self.best_threshold = None
            self.best_index = None
            self.prev_value = prev_value if prev_value else None

        def Tree(self):
            if not self.prev_value:
                raise "Tree didnt proceed no data to split"
                
            elif len(self.x) == 0:
                self.prev_value.Leaf() #Just incase it comes close to make a leaf but fails to be
                return
            
            
            param = [self.x[:,i] for i in range(self.x.shape[1])]
            for n,chunk in enumerate(param):
                GiniList = (
                    []
                )
                for feat in chunk:
                    mask = self.x[:,n] <= feat
                    y_left , y_right = self.y[mask] , self.y[~mask]
                    left_wgini  = self.root.Gini_Adv(y_left , len(self.y))
                    right_wgini = self.root.Gini_Adv(y_right , len(self.y))
                    weighted_gini = left_wgini + right_wgini
                    GiniList.append(weighted_gini)

                if min(GiniList) < self.best_gini:
                    self.best_gini = min(GiniList)
                    self.best_threshold = chunk[np.argmin(GiniList)]
                    self.best_index = n

            if self.best_index == None and self.best_threshold == None:
                self.prev_value.Leaf()#Just incase it comes close to make a leaf but fails to be
                return

            mask = self.x[:,self.best_index] <= self.best_threshold
            x_left , y_left , x_right , y_right = self.x[mask], self.y[mask] , self.x[~mask], self.y[~mask]
            self.prev_value  = self.root.Node(self.root , x_left , y_left , self.prev_value) #incase the recursion ends without a proper near 0.0 gini.

            if round(self.best_gini,2) == 0.0:
                self.left = self.root.Node(self.root , x_left , y_left,self.prev_value)
                self.right = self.root.Node(self.root , x_right , y_right , self.prev_value)
                #cause weighted gini evaluates both side gini and both reach near 0.0 means both dont need to be shifted.
                self.right.Leaf() 
                self.left.Leaf()
                 

            else:
                self.left = self.root.Node(self.root , x_left , y_left , self.prev_value)
                self.right = self.root.Node(self.root , x_right , y_right , self.prev_value)
                self.left.Tree()
                self.right.Tree()

        def Leaf(self):
            val , count = np.unique(self.y , axis= 0 , return_counts=True)
            max_count_index = np.argmax(count)
            return val[max_count_index]



    def Gini_Adv(self , labA:np.array , n):
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
        weight = total/n
        return gini_impurity * weight
        
  