from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO  
from IPython.display import Image  
import matplotlib.pyplot as plt
import pydotplus
from statistics import mean
from six import StringIO
import six
import sys
#sys.modules['sklearn.externals.six'] = six

import os
os.chdir("E://Level Four//First Term//Machine 2//Assignment1")
print("Current Working Directory " , os.getcwd())

col_name=['Class Name ','handicapped-infants','water-project-cost-sharing','adoption-of-the-budget-resolution',
          'physician-fee-freeze','el-salvador-aid','religious-groups-in-schools','anti-satellite-test-ban',
          'aid-to-nicaraguan-contras','mx-missile','immigration','synfuels-corporation-cutback',
          'education-spending','superfund-right-to-sue','crime','duty-free-exports','export-administration-act-south-africa']

VotesData=pd.read_excel(r'house-votes-84.xlsx',header=None,names=col_name )



def HandleMissingValue(Data,LHeader):
    for i in range(1,len(LHeader)):
        ncount=0
        ycount=0
        py=0
        pn=0
        for j in range(len(Data)):
            if Data[LHeader[i]][j]=="n":
                ncount+=1
            elif Data[LHeader[i]][j]=="y":
                ycount+=1
        py=ycount/len(Data)
        pn=ncount/len(Data)
        if py>pn :
            for j in range(len(Data)):
                if Data[LHeader[i]][j]=="?":
                    Data[LHeader[i]][j]='y'
        else:
            for j in range(len(Data)):
                if Data[LHeader[i]][j]=="?":
                    Data[LHeader[i]][j]='n'
    return Data

#to encode y to 1 and n to 0

def EncodesTree(HandledData,LHeader):
    for i in range(1,len(LHeader)):
        for j in range(len(HandledData)):
            if HandledData[LHeader[i]][j]=='y':
                HandledData[LHeader[i]][j]=1
            else :
                HandledData[LHeader[i]][j]=0
    return HandledData

def Decision_Tree_Model(x,y,testsize):
    #Splitting Data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=testsize, random_state=1)
    DTC = DecisionTreeClassifier(criterion="entropy")
    DTC=DTC.fit(X_train,y_train)
    y_pred = DTC.predict(X_test)
    Accurcy=metrics.accuracy_score(y_test, y_pred)
    return DTC ,Accurcy
                
LHeader=list(VotesData.head())


# New data after remove missing values and  encoding it 

NewData=EncodesTree(HandleMissingValue(VotesData,LHeader),LHeader)

#Feature Selection

L=['handicapped-infants', 'water-project-cost-sharing',
   'adoption-of-the-budget-resolution', 'physician-fee-freeze',
   'el-salvador-aid', 'religious-groups-in-schools',
   'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
   'mx-missile', 'immigration', 'synfuels-corporation-cutback',
   'education-spending', 'superfund-right-to-sue', 'crime', 
   'duty-free-exports', 'export-administration-act-south-africa']
L2=LHeader[0]
X = NewData[L] #Features
Y=NewData[L2] # target

#Passing different training set size to show the varity of accuracy 
#coresponding to the size
AccandSize={}
DTCList=[]
i=0.7
No_Nodes=[] 
while i>=0.3:
    DTC,Acc=Decision_Tree_Model(X,Y,i)
    AccandSize[1-i]=Acc
    No_Nodes.append(DTC.tree_.node_count) 
    DTCList.append(DTC)
    i-=(0.1)
    
#Report the sizes and accuracies of these trees in each experiment
f=open("Report.txt",'a')
D,A=Decision_Tree_Model(X,Y,.75)
#f.write("Accuracy of training set 25% reurn 3 times \n")
f.write("Accuracy: "+str(A))
f.write("\n")
f.close()

acc=list(AccandSize.values())
MaxAcc=max(acc)
MinAcc=min(acc)
MeanAcc=mean(acc)

#report the mean, maximum and minimum accuracies 
f2=open("Report2.txt",'a')
#f2.write("Accuracy of training sets varies from 30% to 70% \n")
f2.writelines("Maximum : "+str(MaxAcc)+"\n")
f2.writelines("Minimum : "+str(MinAcc)+"\n")
f2.writelines("Average : "+str(MeanAcc)+"\n\n")
f2.close()


MaxNoNodes=max(No_Nodes)
MinNoNodes=min(No_Nodes)
MeanNodes=mean(No_Nodes)
#Report the Number of Nodes for these trees in each experiment
f3=open("Report3.txt",'a')
#f3.write("No. Nodes of training sets varies from 30% to 70% \n")
f3.write("Maximum No. Nodes: "+str(MaxNoNodes)+"\n")
f3.write("Minimum No. Nodes: "+str(MinNoNodes)+"\n")
f3.write("Average No. Nodes: "+str(MeanNodes)+"\n\n")
f3.close()


ind=acc.index(MaxAcc)
size=list(AccandSize.keys())

#plot showing how accuracy varies with training set size
plt.plot(size,acc, color='green', linestyle='dashed', linewidth = 3,
         marker='o', markerfacecolor='blue', markersize=12)
plt.ylim(0.8,1)
plt.xlim(0,1)

plt.xlabel('Size')

plt.ylabel('Accuracy')

plt.title('Accuracy varies with training set size')

plt.show()

#show  how the number of nodes in 
#the final tree varies with training set size

plt.plot(size,No_Nodes, color='green', linestyle='dashed', linewidth = 3,
         marker='o', markerfacecolor='blue', markersize=12)
plt.ylim(0,50)
plt.xlim(0,1)

plt.xlabel('Size')

plt.ylabel('Number of Nodes')

plt.title('how the number of nodes in the final tree varies with training set size')

plt.show()

#Visualizing Decision Trees

DTC2=DTCList[ind]
dot_data = StringIO()
export_graphviz(DTC2, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,
                feature_names = L,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('United States Congressional Voting Records.png')
Image(graph.create_png())


        

