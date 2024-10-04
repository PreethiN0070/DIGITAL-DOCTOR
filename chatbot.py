import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

training = pd.read_csv('Training.csv')
testing = pd.read_csv('Testing.csv')
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']

reduced_data = training.groupby(training['prognosis']).max()

# Mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

scores = cross_val_score(clf, x_test, y_test, cv=3)
print(scores.mean())

model = SVC()
model.fit(x_train, y_train)
print("For SVM:")
print(model.score(x_test, y_test))

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

severityDictionary = {}
description_list = {}
precautionDictionary = {}

symptoms_dict = {symptom: index for index, symptom in enumerate(x)}
global symptom_index
symptom_index = 0

def readn(nstr):
    engine = pyttsx3.init()
    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)
    engine.say(nstr)
    engine.runAndWait()
    engine.stop()


def calc_condition(exp, days):
    sum_severity = 0
    count_symptoms = 0
    for item in exp:
        if item in severityDictionary:
            sum_severity += severityDictionary[item]
            count_symptoms += 1
        else:
            return f"Severity for symptom '{item}' not found in the database."

    if count_symptoms == 0:
        return "No severity information found for any of the symptoms entered."
    else:
        if ((sum_severity * days) / count_symptoms) > 13:
            return "You should take the consultation from a doctor."
        else:
            return "It might not be that bad, but you should take precautions."



def getDescription():
    global description_list
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            description_list[row[0]] = row[1]


def getSeverityDict():
    global severityDictionary
    with open('symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) >= 2:  # Check if row has at least 2 elements
                severityDictionary[row[0]] = int(row[1])


def getprecautionDict():
    global precautionDictionary
    with open('symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]


def getInfo(user_name=None):
    if user_name:
        return f"Hello, {user_name}! Enter the symptom you are experiencing."
    else:
        return "Send a greeting message first (hi, hello, hey)."


def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return 0, []


def sec_predict(symptoms_exp):
    df = pd.read_csv('Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])


def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))


def tree_to_code(tree, feature_names, num_days, disease_input, current_symptom_index):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = feature_names
    symptoms_present = []
    
    def recurse(node, depth):
        nonlocal symptoms_present
        global symptom_index
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                return recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                return recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            print("Else####################################")
            response_message = "Are you experiencing any symptoms? "
            print(symptoms_given)
            print(symptom_index)
            if symptom_index != len(symptoms_given):
                print("####################SYMPTOMS#########################")
                i = symptom_index
                symptom_index += 1
                return (response_message,symptoms_given[i])
            # for symptom in symptoms_given:
            #     response_message += f"{symptom} ? : \n"

            second_prediction = sec_predict(symptoms_given)
            calc_condition(symptoms_given, num_days)
            if present_disease[0] == second_prediction[0]:
                response_message = f"You may have {present_disease[0]}."
                response_message += f"{description_list[present_disease[0]]}"
            else:
                response_message += f"You may have {present_disease[0]} or {second_prediction[0]}.\n"
                response_message += f"{description_list[present_disease[0]]}\n"
                response_message += f"{description_list[second_prediction[0]]}\n"

            precution_list = precautionDictionary[present_disease[0]]
            response_message += "Take following measures:"
            for i, j in enumerate(precution_list):
                response_message += f"{i + 1}) {j}"

            return response_message

    return recurse(0, 1)



def main():
    getDescription()
    getSeverityDict()
    getprecautionDict()
    print(getInfo())
    num_days = int(input("From how many days? : "))
    disease_input = input("Enter the symptom you are experiencing: ")
    response = tree_to_code(clf, cols, num_days)
    print(response)
    print("----------------------------------------------------------------------------------------")


if __name__ == '__main__':
    main()
