import re
import pandas as pd
import pyttsx3
import csv
import warnings
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.tree import plot_tree , _tree
from sklearn.ensemble import RandomForestClassifier
from svc_chatbot import start_chatting

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    training = pd.read_csv('Data/Training.csv')
    testing = pd.read_csv('Data/Testing.csv')
    cols = training.columns
    cols = cols[:-1]
    x = training[cols]
    y = training['prognosis']
    y1 = y

    reduced_data = training.groupby(training['prognosis']).max()

    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=0)
    testx = testing[cols]
    testy = testing['prognosis']
    testy = le.transform(testy)

    # plt.figure(figsize=(200,200))  # Adjust the figure size as needed
    # plot_tree(clf, filled=True, feature_names=cols, class_names=le.classes_)
    # plt.show()

    severityDictionary = {}
    description_list = {}
    precautionDictionary = {}
    symptoms_dict = {}

    def readn(nstr):
        engine = pyttsx3.init()

        engine.setProperty('voice', "english+f5")
        engine.setProperty('rate', 200)

        engine.say(nstr)
        engine.runAndWait()
        engine.stop()

    def calc_condition(exp,days):
        sum=0
        for item in exp:
            if item in severityDictionary:
                sum=sum+severityDictionary[item]
        if((sum*days)/(len(exp)+1)>13):
            print("\nYou should take the consultation from doctor. ")
        else:
            print("\nIt might not be that bad but you should take precautions.")


    def getDescription():
        with open('MasterData/symptom_Description.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                _description={row[0]:row[1]}
                description_list.update(_description)

    def getSeverityDict():
        global severityDictionary
        with open('MasterData/symptom_severity.csv') as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            try:
                for row in csv_reader:
                    _diction={row[0]:int(row[1])}
                    severityDictionary.update(_diction)
            except:
                pass


    def getprecautionDict():
        precautionDictionary
        with open('MasterData/symptom_precaution.csv') as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                _prec={row[0]:[row[1],row[2],row[3],row[4]]}
                precautionDictionary.update(_prec)


    def getInfo():
        print("-----------------------------------HealthCare ChatBot-----------------------------------")
        print("Please enter the following information : ")
        print("\nName : ",end="->")
        name=input("")
        print("Age : ",end="-> ")
        age=input("")
        print("Gender : ",end="-> ")
        gender=input("")
        print("Medical History : ",end="-> ")
        medical_history=input("")
        print("\nHello, ",name,"!")

    def check_pattern(dis_list,inp):
        pred_list=[]
        patterns = [pattern.strip().replace(' ', '_') for pattern in inp.split(',')]
        combined_pattern = '|'.join(patterns)
        regexp = re.compile(combined_pattern)
        pred_list=[item for item in dis_list if regexp.search(item)]
        if(len(pred_list) < 4):
            print("Please enter a minimum of 4 valid symptoms for accurate diagnosis")
            return -1,[]
        if(len(pred_list)>0):
            return 1,pred_list
        else:
            return 0,[]

    def sec_predict(symptoms_exp):
        df = pd.read_csv('Data/Training.csv')
        X = df.iloc[:, :-1]
        y = df['prognosis']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)

        rf_clf = RandomForestClassifier(n_estimators=50, random_state=20)

        rf_clf.fit(X_train, y_train)

        symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
        input_vector = np.zeros(len(symptoms_dict))
        for item in symptoms_exp:
            input_vector[[symptoms_dict[item]]] = 1

        return rf_clf.predict([input_vector])


    def print_disease(node):
        node = node[0]
        val  = node.nonzero() 
        disease = le.inverse_transform(val[0])
        return list(map(lambda x:x.strip(),list(disease)))

    def tree_to_code():
        chk_dis=cols
        feature_names = cols
        symptoms_present = []
        parent_nodes = []

        while True:

            print("\nEnter the (3-4) main symptoms that you are experiencing for accurate diagnosis \t",end="->")
            disease_input = input("").lower()
            conf,cnf_dis=check_pattern(chk_dis,disease_input)
            if conf==1:
                if(len(cnf_dis) > 0):
                    print("Please select the symtpoms by entering their ids(0 ,1 ,2..etc): ")
                    for num,it in enumerate(cnf_dis):
                        print(num,")",it)
                    if num!=0:
                        print(f"Confirm the ones you meant (0 - {num}):  ", end="")
                        conf_input = input("").lower()
                        conf_inp = [int(x) for x in conf_input.split(',')]
                    else:
                        conf_inp=[0]
                disease_input_list = []
                
                if len(conf_inp) <= len(cnf_dis):
                    for conf_inps in conf_inp:
                        if conf_inps < len(cnf_dis):
                            disease_input_list.append(cnf_dis[conf_inps])
                else:
                    print("Oops! We don't have much information related to that symptom. Please visit a doctor near you for more information.")
                    return
                break
            elif conf == -1:
                continue
            else:
                print("Sorry we don't have much information on that symptom. Try a different symptom?")

        while True:
            try:
                num_days=int(input("How many days have you been experiencing this symptom ? : "))
                break
            except:
                print("Enter valid input.")

        def recurse(node, depth, tree_, feature_name):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]

                if name in disease_input_list:
                    val = 1
                else:
                    val = 0

                if  val <= threshold:
                    recurse(tree_.children_left[node], depth + 1, tree_, feature_name)
                else:
                    symptoms_present.append(name)
                    parent_nodes.append(tree_.children_right[node])
                    recurse(tree_.children_right[node], depth + 1, tree_, feature_name)
            
            return parent_nodes, symptoms_present

        def checkMoreFeatures(node, symptoms_present, tree_):
                present_disease = print_disease(tree_.value[node])
                print( "You may have " , present_disease)
                red_cols = reduced_data.columns 
                symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
                dis_list=list(symptoms_present)
                # if len(dis_list)!=0:
                #     print("symptoms present  " + str(list(symptoms_present)))
                # print("symptoms given "  +  str(list(symptoms_given)) )
                print("In addition to that, are you experiencing any of the following symptoms : ")
                symptoms_exp= [] + disease_input_list
                for syms in list(symptoms_given):
                    if syms not in disease_input_list:
                        inp=""
                        print(syms,"? : ",end='')
                        while True:
                            inp=input("").lower()
                            if(inp=="yes" or inp=="no"):
                                break
                            else:
                                print("Please answer yes/no : ",end="")
                        if(inp=="yes"):
                            symptoms_exp.append(syms)

                second_prediction=sec_predict(symptoms_exp)
                print(second_prediction)

                calc_condition(symptoms_exp,num_days)
                if(len(present_disease) > 0 and len(second_prediction) > 0 and present_disease[0]==second_prediction[0]):
                    print("You may have ", present_disease[0])
                    print(description_list[present_disease[0]])

                    readn(f"You may have {present_disease[0]}")
                    readn(f"{description_list[present_disease[0]]}")

                else:
                    print("You may have ", present_disease[0], "or ", second_prediction[0])
                    if len(description_list) > 0:
                        print(description_list[present_disease[0]])
                    if len(second_prediction) > 0:
                        print(description_list[second_prediction[0]])
                precution_list=precautionDictionary[present_disease[0]]
                print("Take following measures : ")
                for  i,j in enumerate(precution_list):
                    print(i+1,")",j)

        

        df = pd.read_csv('Data/Training.csv')
        X = df.iloc[:, :-1]
        y = df['prognosis']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
        rf_clf = RandomForestClassifier(n_estimators=50, random_state=20)  # You can adjust the number of estimators as needed
        # Fit Random Forest classifier to training data
        rf_clf.fit(X_train, y_train)
        node_tree_symptom_mapping = {}
        for i, tree in enumerate(rf_clf.estimators_):
            tree_ = tree.tree_
            feature_name = [
                feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                for i in tree_.feature
            ]
            symptoms_present = []
            returned_parent_nodes, confirm_symptoms_present = recurse(0,1, tree_, feature_name)
            if(len(returned_parent_nodes) != 0):
                node_tree_symptom_mapping[(returned_parent_nodes[-1], tree_)] = confirm_symptoms_present
        
        symptoms_lengths = {item: len(symptoms) for item, symptoms in node_tree_symptom_mapping.items()}
        if len(symptoms_lengths) > 0:
            item_with_longest_symptoms = max(symptoms_lengths, key=symptoms_lengths.get)
            longest_symptoms_present = node_tree_symptom_mapping[item_with_longest_symptoms]
            checkMoreFeatures(item_with_longest_symptoms[0], longest_symptoms_present, item_with_longest_symptoms[1])
        else:
            print("Sorry! We don't have much information on that symptom.")

        
    for index, symptom in enumerate(x):
        symptoms_dict[symptom] = index

    
    getSeverityDict()
    getDescription()
    getprecautionDict() 
    getInfo()
    tree_to_code()
    start_chatting()

    print("----------------------------------------------------------------------------------------")
    

if __name__ == "__main__":
    main()

