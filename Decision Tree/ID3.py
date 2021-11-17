import numpy as np
import math
import os
import pprint



class id3tree():

    def __init__(self, dataset_loc):
        self.dataset = np.genfromtxt(dataset_loc, delimiter=',', dtype="|U5")
        self.datasetLen = len(self.dataset)
        train_n = int(0.8*self.datasetLen)
        self.attributes = self.dataset[0, :-1]
        trainIndices = []
        p = [z for z in range(self.datasetLen)]
        plen = self.datasetLen
        for i in range(train_n):
            j = np.random.randint(low=0,high=plen)
            x=p.pop(j)
            trainIndices.append(x)
            plen -= 1
        self.trainData = np.array([self.dataset[n] for n in trainIndices])
        self.testData = [self.dataset[r] for r in p]
        self.targetIndex = len(self.attributes)-1
        self.attIndexes = [i for i in range(len(self.attributes))]
        self.tarClasses = np.unique(self.trainData)
        self.root = ""
    
    def findbest(self, examples, atts):
        targlist = examples[:,-1]
        entA = self.calcEnt(targlist)
        attEnts = []
        exampleLen = len(examples)
        for att in atts:
            attlist = examples[:,att]
            attVals, attValLen = np.unique(attlist, return_counts=True)
            attIndex = []
            for j in attVals:
                pieceOfShit = np.where(attlist==j)[0]
                attIndex.append(pieceOfShit)
            attEnt = 0
            for k in range(len(attVals)):
                attTgt_k = [targlist[l] for l in attIndex[k]]
                entV_k = self.calcEnt(attTgt_k)
                attEnt += (attValLen[k]/exampleLen)*entV_k
            attEnts.append(attEnt)
        finEntList = [entA-attEnts[m] for m in range(len(atts))]
        bestAttId = finEntList.index(max(finEntList))
        bestAtt = atts[bestAttId]
        return bestAtt
            
                
    def calcEnt(self, examples):
        counts = np.unique(examples, return_counts=True)[1]
        totalCounts = np.sum(counts)
        entropy = 0
        for i in counts:
            entropy -= (i/totalCounts)*math.log2(i/totalCounts)
        return entropy
    

    def trainTree(self, examples, attributes, depth, tree=None):
        if depth==3:
            clvals,clcounts = np.unique(examples[:,-1], return_counts=True)
            maxId = np.argmax(clcounts)
            return clvals[maxId]

        node = self.findbest(examples, attributes)

        if tree is None:
            tree = {}
            tree[node] = {}
        
        attVals = np.unique(examples[:,node])
        
        for val in attVals:
            valIndexes = np.where(examples[:,node]==val)[0]
            subtable = np.array([examples[i] for i in valIndexes])
            clvalue, counts = np.unique(subtable[:,-1], return_counts=True)

            if len(counts)==1:
                tree[node][val] = clvalue[0]
            else:
                print("deleting node: {} \n".format(node))
                print("from list: {} \n".format(attributes))
                nodeid = np.where(attributes==node)[0]
                newattrs = np.delete(attributes, nodeid)
                print("the new list is: {} \n".format(newattrs))
                newdepth = depth+1
                tree[node][val] = self.trainTree(subtable, newattrs, newdepth)
        
        return tree
    
    def justdoit(self):
        self.finaltree = self.trainTree(self.trainData, self.attIndexes, -1)
        return self.finaltree

    def testid3(self):
        n = 0
        nt = len(self.testData)
        for i in range(nt):
            arr = self.testData[i]
            key = self.find_ans(self.finaltree, arr)
            if(key==arr[-1]):
                n += 1
        
        self.accuracy = n/nt * 100
        return self.accuracy


            
    def find_ans(self, tree, arr):
        for j in tree.keys():
            for k in tree[j].keys():
                if(arr[j]==k):
                    if(isinstance(tree[j][k],str)):
                        return tree[j][k]
                    elif(isinstance(tree[j][k],dict)):
                        return self.find_ans(tree[j][k],arr)

dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prison_dataset.csv")
newid3 = id3tree(dataset_path)
id3trained = newid3.justdoit()
pprint.pprint(id3trained)
id3accuracy = newid3.testid3()
print(id3accuracy)

            




        
            

