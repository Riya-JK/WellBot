import re 
import pandas as pd 
import pyttsx3 
import csv
import warnings 
from sklearn import preprocessing 
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.tree import plot_tree, _tree 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
from mood_analyzer import start_mood_analysis

warnings.filterwarnings("ignore", category=DeprecationWarning)  # Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress UserWarnings

description_list = {}  # Initialize dictionary to store symptom descriptions
precautionDictionary = {}  # Initialize dictionary to store precautions for diseases
disease_to_speciality_mapping = {}  # Initialize dictionary to map diseases to medical specialties

# Function to save model if accuracy is above threshold
def save_model_if_accuracy_above_threshold(model, accuracy, threshold, filename):
    if accuracy > threshold:
        joblib.dump(model, filename)  # Save the model
    #     print("Model saved successfully.")

# Function to evaluate classifier performance
def evaluate_classifier(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    precision = precision_score(y_test, y_pred, average="weighted")  # Calculate precision
    recall = recall_score(y_test, y_pred, average="weighted")  # Calculate recall
    return accuracy, precision, recall

def main():
    # Read training and testing datasets
    training = pd.read_csv('Disease_Dataset/Training.csv')
    testing = pd.read_csv('Disease_Dataset/Testing.csv')
    cols = training.columns
    cols = cols[:-1]
    x = training[cols]
    y = training['prognosis']
    
    # Group training data by prognosis and select maximum values
    reduced_data = training.groupby(training['prognosis']).max()

    # Encode target labels
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=0)
    testx = testing[cols]
    testy = testing['prognosis']
    testy = le.transform(testy)

    severityDictionary = {}  # Dictionary to store symptom severity
    precautionDictionary = {}  # Dictionary to store precautions for diseases
    symptoms_dict = {}  # Dictionary to map symptoms to indices

    # Function to load doctors' data
    def load_doctors_data():
        global doctors_data
        try:
            doctors_data = pd.read_csv('Symptom_MasterData/doctors.csv', encoding='utf-8')
        except UnicodeDecodeError:
            doctors_data = pd.read_csv('Symptom_MasterData/doctors.csv', encoding='ISO-8859-1')

    # Function to read symptom descriptions from CSV
    def getDescription():
        global description_list, disease_to_speciality_mapping
        with open('Symptom_MasterData/symptom_Description.csv', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None)  # Skip the header
            for row in csv_reader:
                if len(row) >= 3:
                    disease = row[0].strip().lower()  # Disease name
                    _description = {row[0]: (row[1], row[2])}  # Symptom description and specialty
                    specialty = row[2].strip()  # Specialty
                    description_list.update(_description)
                    disease_to_speciality_mapping[disease] = specialty
                else:
                    print("Row format error, expected at least 3 columns:", row)

    # Function to get doctor recommendations based on specialty
    def getDoctorRecommendations(specialty):
        matches = doctors_data[doctors_data['speciality'].str.lower() == specialty.lower()]
        if not matches.empty:
            recommendations = matches.head(3)  # Limit to top 3 matches for brevity
            recommendations_info = recommendations.apply(
                lambda row: f"{row['Doctor\'s Name']} - Specialty: {row['speciality']}, More Info: {row['Link']}",
                axis=1
            ).tolist()
            recommendation_str = "\n".join(recommendations_info)
            return f"\nFor the specialty {specialty}, the following doctors are recommended:\n{recommendation_str}"
        return "No specialist recommendation found for this specialty."

    # Function to get severity dictionary from CSV
    def getSeverityDict():
        global severityDictionary
        with open('Symptom_MasterData/symptom_severity.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            try:
                for row in csv_reader:
                    _diction = {row[0]: int(row[1])}  # Symptom severity
                    severityDictionary.update(_diction)
            except:
                pass

    # Function to get precaution dictionary from CSV
    def getprecautionDict():
        global precautionDictionary
        with open('Symptom_MasterData/symptom_precaution.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if len(row) == 6:  # Check if the link column exists
                    _prec = {row[0]: [row[1], row[2], row[3], row[4], row[5]]}  # Precautions with link
                else:
                    _prec = {row[0]: [row[1], row[2], row[3], row[4]]}  # Precautions without link
                precautionDictionary.update(_prec)

    # Function to get user information
    def getInfo():
        global medical_history
        print("-----------------------------------HealthCare ChatBot-----------------------------------")
        print("Please enter the following information:")
        print("\nName: ", end="->")
        name = input("")
        print("Age: ", end="-> ")
        age = input("")
        print("Gender: ", end="-> ")
        gender = input("")
        print("Medical History: ", end="-> ")
        medical_history = input("")
        print("\nHello,", name, "!")

    # Function to check symptom pattern
    def check_pattern(dis_list, inp):
        pred_list = []
        patterns = [pattern.strip().replace(' ', '_') for pattern in inp.split(',')]
        combined_pattern = '|'.join(patterns)
        regexp = re.compile(combined_pattern)
        for item in dis_list:
            if regexp.search(item):
                pred_list.append(item)
        if len(pred_list) < 3:
            print("\nPlease enter a minimum of 3 valid symptoms for accurate diagnosis")
            return -1, []
        if len(pred_list) > 0:
            return 1, pred_list
        else:
            return 0, []

    # Function for secondary prediction
    def sec_predict(symptoms_exp):
        df = pd.read_csv('Disease_Dataset/Training.csv')
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

    # Function to print disease based on decision tree nodes
    def print_disease(node):
        node = node[0]
        val = node.nonzero()
        disease = le.inverse_transform(val[0])
        return list(map(lambda x: x.strip(), list(disease)))

    # Function to ask the user to confirm any additional symptoms that they are experiencing and make second prediction
    def tree_to_code():
        chk_dis = cols
        feature_names = cols
        symptoms_present = []
        parent_nodes = []

        while True:
            print("\nEnter the (3-4) main symptoms that you are experiencing for accurate diagnosis\t", end="->")
            disease_input = input("").lower()
            conf, cnf_dis = check_pattern(chk_dis, disease_input)
            if conf == 1:
                if len(cnf_dis) > 0:
                    print("Please select the symptoms by entering their ids (0, 1, 2, etc.): ")
                    for num, it in enumerate(cnf_dis):
                        print(num, ")", it)
                    if num != 0:
                        print(f"Confirm the ones you meant (0 - {num}):  ", end="")
                        conf_input = input("").lower()
                        if conf_input.endswith(","):
                            conf_input = conf_input[:-1]
                        conf_inp = [int(x) for x in conf_input.split(',')]
                    else:
                        conf_inp = [0]
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
                print("Sorry, we don't have much information on that symptom. Try a different symptom?")

        while True:
            try:
                num_days = int(input("How many days have you been experiencing this symptom? : "))
                break
            except:
                print("Enter valid input.")

        #function to recursively traverse through each decision tree in th random forest to fetch the trees with most symptom overlap
        def recurse(node, depth, tree_, feature_name):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]

                if name in disease_input_list:
                    val = 1
                else:
                    val = 0

                if val <= threshold:
                    recurse(tree_.children_left[node], depth + 1, tree_, feature_name)
                else:
                    symptoms_present.append(name)
                    parent_nodes.append(tree_.children_right[node])
                    recurse(tree_.children_right[node], depth + 1, tree_, feature_name)

            return parent_nodes, symptoms_present

        #check if more symptoms match to make a second prediction
        def checkMoreFeatures(node, symptoms_present, tree_):
                present_disease = print_disease(tree_.value[node])
                red_cols = reduced_data.columns 
                symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
                dis_list = list(symptoms_present)
                print("In addition to that, are you experiencing any of the following symptoms:")
                symptoms_exp = [] + disease_input_list
                for syms in list(symptoms_given):
                    if syms not in disease_input_list:
                        inp = ""
                        print(syms, "? : ", end='')
                        while True:
                            inp = input("").lower()
                            if(inp == "yes" or inp == "no"):
                                break
                            else:
                                print("Please answer yes/no : ", end="")
                        if(inp == "yes"):
                            symptoms_exp.append(syms)

                second_prediction = sec_predict(symptoms_exp)

                calc_condition(symptoms_exp, num_days)
                if(len(present_disease) > 0 and len(second_prediction) > 0 and present_disease[0] == second_prediction[0]):
                    print("You may have", present_disease[0])
                    if present_disease[0] in description_list:
                        print(description_list[present_disease[0]][0])
                        readn(f"You may have {present_disease[0]}")
                        readn(f"{description_list[present_disease[0]][0]}") 
                    else:
                        print("No description found")

                else:
                    print("You may have ", present_disease[0], "or ", second_prediction[0])
                    if len(description_list) > 0:
                        print(description_list[present_disease[0]][0])
                    if len(second_prediction) > 0:
                        print(description_list[second_prediction[0]][0])

                matched_with_medical_history = False

                present_disease_words = present_disease[0].lower().split()
                second_prediction_words = second_prediction[0].lower().split()
                medical_history_words = medical_history.lower().split()

                for word in present_disease_words + second_prediction_words:
                    if word in medical_history_words:
                        matched_with_medical_history = True
                        break

                if matched_with_medical_history:
                    print("Your medical condition might be related to your past medical history.")
                else:
                    print("You don't have anything related to your past medical history.")

                precution_list = precautionDictionary[present_disease[0]]
                print("\nTake following measures:")
                for  i, j in enumerate(precution_list):
                    print(i+1, ")", j)

                if type(present_disease) == list:
                    specialty = disease_to_speciality_mapping.get(present_disease[0].lower(), "Unknown")
                else:
                    specialty = disease_to_speciality_mapping.get(present_disease.lower(), "Unknown")
                doctor_recommendations = getDoctorRecommendations(specialty)
                if present_disease[0].lower() in description_list:
                    print(description_list[present_disease[0]][1].lower())

                print(doctor_recommendations)
                print("\nLoading your wellness assistant ....")

        df = pd.read_csv('Disease_Dataset/Training.csv')
        X = df.iloc[:, :-1]
        y = df['prognosis']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
        rf_clf = RandomForestClassifier(n_estimators=50, random_state=20)  # You can adjust the number of estimators as needed
        rf_clf.fit(X_train, y_train)

        node_tree_symptom_mapping = {}
        for i, tree in enumerate(rf_clf.estimators_): #iterate over each tree in the random forest
            tree_ = tree.tree_
            feature_name = [
                feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                for i in tree_.feature
            ]
            symptoms_present = []
            returned_parent_nodes, confirm_symptoms_present = recurse(0, 1, tree_, feature_name)
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

    load_doctors_data()
    getSeverityDict()
    getDescription()
    getprecautionDict()
    getInfo()
    tree_to_code()
    start_mood_analysis()

    print("----------------------------------------------------------------------------------------")

if __name__ == "__main__":
    main()
