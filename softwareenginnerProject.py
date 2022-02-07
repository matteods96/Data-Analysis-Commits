# -*- coding: utf-8 -*-
"""
Created on Sun May 19 15:10:47 2019

@author: Matteo
"""

import pandas as pd#use shorthand pd for  the pandas library
import numpy as np#use shorthand np for  the numpy library
from sklearn.model_selection import train_test_split  
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor 
from IPython.display import Image 






def calculateEvenness(filename,project_id):
    df=pd.read_csv(filename,delimiter=";")#Load the dataframe reading the file commits.csv
    df['time'] = df['time'].apply(lambda x: int((x/24)/7))#converts hours into weeks in the dataframe 
    values=df[['project_id','time']]
    x= values[values.project_id==project_id ]
    #In Ipython console it appear that x containts [401 rows x 2 columns]
    total_commits=x.shape[0]#total commits
    print("Number of total commits  is:",total_commits)
    # approach of finding the number of weeks
    z=df['time'].tolist()
    #the variable z is the list of values in the column time in the given project_id
    total_weeks=max(z)-min(z)
    #the number of weeks it is defined as  the difference between the max of the week and the min of the week
    #print()
#   print("The number of weeks is:",total_weeks)
    #print("The total number of weeks:", total_weeks)
    max_var_values=list(np.zeros(total_weeks-1, dtype=int))
    max_var_values.append(total_commits)
    print(max_var_values)
    #print(max_var_values)
    max_variance=np.var(max_var_values)
    print("The maximum variance is:",max_variance)
    #print(y)
    y=x['time'].tolist()
    commits_per_week, edges = np.histogram(y, bins=np.arange(0, 1+total_weeks))
    actual_variance=np.var(commits_per_week)
    #print("The actual variance is:",actual_variance)
    evennessScore = 1-(actual_variance/max_variance)
    print("Evenness Score for the  project_id is",evennessScore)
    return evennessScore


def calculateLateness(filename,project_id):
    df=pd.read_csv(filename,delimiter=";")#Load the dataframe reading the file commits.csv
    df['time'] = df['time'].apply(lambda x: int((x/24)/7))#converts hours into weeks in the dataframe 
    values=df[['project_id','time']]
    x= values[values.project_id==project_id ]
    total_commits=x.shape[0]#total commits
    #print("Number of total commits  is:",total_commits)
    # approach of finding the number of weeks
    z=x['time'].tolist()#the variable z is the list of values in the column time inthe given project_id
    number_of_weeks=max(z)-min(z)
    weights=np.linspace(start=0, stop=1, num=number_of_weeks)
    lateness=0
    for i in range(0,len(weights)):
        value=z[i]/total_commits*weights[i]
        lateness+=value
    
    return lateness
     
def calculateEarlyness(filename,project_id):
     return 1-calculateLateness(filename,project_id)  



def createNewDataset(filename):
     df=pd.read_csv(filename,delimiter=";")#Load the dataframe reading the file commits.csv
     data= pd.read_json("C:\\Users\\Matteo\\Downloads\\grades.json", orient='columns') 
     a=list(set(df['project_id']))
     print(a)
     row=[]
     print(len(a))
     new_data = data.sort_values('project_id')
     grades=new_data['grade'].tolist()
     del grades[16]#delete the grades of the project 17 as this does not exist in the file commits.csv
     print(len(grades))
     #print(new_data)
     header = ['project_id', 'evenness', 'lateness','earlyness','grade']
     lines = [[]] * len(a)
     #print(lines)
     for i in range(0,len(a)):
         row=[a[i],calculateEvenness(filename,a[i]),calculateLateness(filename,a[i]),calculateEarlyness(filename,a[i]),grades[i]]
         lines.append(row)
     print(lines)
     with open("C:\\Users\\Matteo\\Documents\commits_newdataset.csv", "w", newline='') as f:
          writer = csv.writer(f, delimiter=';')
          writer.writerow(header) # write the header
          # write the actual content line by line
          for l in lines:
              writer.writerow(l)
     f.close()  


def createTreePredictingGrades():# Predicting grade using a regression tree
     df=pd.read_csv("C:\\Users\\Matteo\\Documents\commits_newdataset.csv",delimiter=";")#read the dataset 
     # Split the dataset vertically, so that we have the column we are predicting in y and the data in X 
     y = df['grade']
     print(y.head())
     X = df.drop('grade', axis=1) 
     X = X.drop('project_id', axis=1) 
     print(X.head()) 
     # We split the dataset horizontally. We use 20% for testing and 80% for training
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
     # Grow tree, using max_depth and/or min_impurity_decrease (mse, Mean Squared Error)
     regr= DecisionTreeRegressor(max_depth=6)  
     regr.fit(X_train, y_train)  
     # Test tree
     y_pred = regr.predict(X_test) 
     print(y_pred)
     print("Testing score: " + format(regr.score(X_test,y_test)))
     # Display tree
     # Display tree
     #dotfile = open("dtree2.dot", 'w+')
     dotfile = open("./dtree2.dot", 'w') 
     tree.export_graphviz(regr, out_file ="dtree2.dot" , feature_names = X.columns)
     dotfile.close()  
     Image("dtree2.png")


def calculateContributionForAMember(filename,project_id,member):
    df=pd.read_csv(filename,delimiter=";")#Load the dataframe reading the file commits.csv
   # select the values of datafrane  to use Boolean indexing.
    total_additions=df.loc[(df['project_id']==project_id)&(df['committer']==member)  , 'additions'].sum()
    total_deletions=df.loc[(df['project_id']==project_id)&(df['committer']==member), 'deletions'].sum()
    print("User",member)
    print("Number of additions",total_additions)
    print("Number of deletions",total_deletions)
    return total_additions+total_deletions
     
    
    
    
    
def calculateEvenContribution(filename,project_id): # amount of work per member    
       df=pd.read_csv(filename,delimiter=";")#Load the dataframe reading the file commits.csv
       values=df[['project_id','committer','additions','deletions']]
       x=values[values.project_id==project_id ]
       print(x['committer'])
       members=list(set(x['committer']))#list of members in a team
       works=[]#list wich  contains the values of the amount of work done by each member of the team
       for i in range(0,len(members)):
           works.append(calculateContributionForAMember(filename,project_id,members[i]))
       total_contribution=sum(works)
       print("Toal contribution is :",total_contribution)
       max_var_values=list(np.zeros(len(members)-1, dtype=int))
       print(max_var_values)
       max_var_values.append(total_contribution)
       print(max_var_values)
       max_variance=np.var(max_var_values)
       print("The maximum variance is:",max_variance)
       actual_variance=np.var(works)
       print("The actual variance is:",actual_variance)
       print()
       print(max_variance-actual_variance>0)
       return  1-(actual_variance/max_variance)
   
def createNewDataset2(filename):
     df=pd.read_csv(filename,delimiter=";")#Load the dataframe reading the file commits.csv
     data= pd.read_json("C:\\Users\\Matteo\\Downloads\\grades.json", orient='columns') 
     a=list(set(df['project_id']))
     print(a)
     row=[]
     print(len(a))
     new_data = data.sort_values('project_id')
     grades=new_data['grade'].tolist()
     del grades[16]#delete the grades of the project 17 as this does not exist in the file commits.csv
     print(len(grades))
     #print(new_data)
     header = ['project_id', 'evenness', 'lateness','earlyness','even_contribution','grade']
     lines = [[]] * len(a)
     #print(lines)
     for i in range(0,len(a)):
         row=[a[i],calculateEvenness(filename,a[i]),calculateLateness(filename,a[i]),calculateEarlyness(filename,a[i]),calculateEarlyness(filename,a[i]),calculateEvenContribution(filename,a[i]),grades[i]]
         lines.append(row)
     print(lines)
     with open("C:\\Users\\Matteo\\Documents\commits_newdataset2.csv", "w", newline='') as f:
          writer = csv.writer(f, delimiter=';')
          writer.writerow(header) # write the header
          # write the actual content line by line
          for l in lines:
              writer.writerow(l)
     f.close() 
    
def createTreePredictingGrades2():# Predicting grade using a regression tree
     df=pd.read_csv("C:\\Users\\Matteo\\Documents\commits_newdataset2.csv",delimiter=";")#read the dataset - 
     #print(df)
     # Split the dataset vertically, so that we have the column we are predicting in y and the data in X 
     y = df['grade']
     print(y.head())
     X = df.drop('grade', axis=1) 
     X = X.drop('project_id', axis=1) 
     print(X.head()) 
     # We split the dataset horizontally. We use 20% for testing and 80% for training
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
     # Grow tree, using max_depth and/or min_impurity_decrease (mse, Mean Squared Error)
     regr = DecisionTreeRegressor(max_depth=6)  
     regr.fit(X_train, y_train)  
     # Test tree
     y_pred = regr.predict(X_test) 
     print(y_pred)
     print("Testing score: " + format(regr.score(X_test,y_test)))
     # Display tree
     # Display tree
     #dotfile = open("dtree2.dot", 'w+')
     dotfile = open("./dtree3.dot", 'w') 
     tree.export_graphviz(regr, out_file ="dtree3.dot" , feature_names = X.columns)
     dotfile.close()  
     Image("dtree3.png")
       
       
       
        
#calculateEvenness("C:\\Users\\Matteo\\Documents\commits-even.csv",1)
#print(calculateEvenness("C:\\Users\\Matteo\\Documents\commits.csv",32))
#print(calculateLateness("C:\\Users\\Matteo\\Documents\commits.csv",32))
print()
#print(calculateEarlyness("C:\\Users\\Matteo\\Documents\commits.csv",32))
#createNewDataset("C:\\Users\\Matteo\\Documents\commits.csv")
createTreePredictingGrades()
#print(calculateEvenContribution("C:\\Users\\Matteo\\Documents\commits.csv",32))
#createNewDataset2("C:\\Users\\Matteo\\Documents\commits.csv")
#calculateContributionForAMember("C:\\Users\\Matteo\\Documents\commits.csv",32,221)
createTreePredictingGrades2()







