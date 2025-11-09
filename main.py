# ML Algorithm import
import joblib
import readDataset
from model import AdaBoostClassifier, VotingClassifier, KNN
from sklearn.metrics import make_scorer, accuracy_score, classification_report, confusion_matrix, precision_score, roc_auc_score
from readDataset import x_test, x_train, y_train, y_test, x_train_cv, x_test_cv, showFrequencyCountDiagram, showEmotionCountDiagram, showTrainEmotionCountDiagram, showTestEmotionCountDiagram
import readTweets
import seaborn as sns
import matplotlib.pyplot as plt

def saveModel(model, filename):
    joblib.dump(model, filename)

#saveModel(AdaBoostClassifier.adaBoost_model(), "adaBoost.pkl")

def loadModel(filename):
    return joblib.load(filename)

def runModel(filename):
    model = loadModel(filename)
    y_pred = model.predict(x_test_cv)
    printAndShowDiagram(y_test, y_pred, model)
    return model

def runModel_To_Test(model):
    y_pred = model.predict(readTweets.x_cv)
    printAndShowDiagram(readTweets.y, y_pred, model)
    return model

def compareModel(filename1, filename2):
    model1 = loadModel(filename1)
    model2 = loadModel(filename2)

    y_pred1 = model1.predict(x_test_cv)
    y_pred2 = model2.predict(x_test_cv)

    model1_acc = accuracy_score(y_test, y_pred1)
    model2_acc = accuracy_score(y_test, y_pred1)

    if(model1_acc > model2_acc):
        return model1
    else:
        return model2

def printAndShowDiagram(y_test, y_pred, model):
    print(model)
    print(classification_report(y_test, y_pred))
    print("Accuracy: " + str(accuracy_score(y_test, y_pred)))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, xticklabels=['predicted_anger', 'predicted_fear', 'predicted_joy', 'predicted_sad'],
                yticklabels=['actual_anger', 'actual_fear', 'actual_joy', 'actual_sad'],
                annot=True, fmt='d', annot_kws={'fontsize': 25}, cmap="YlGnBu");
    plt.title(model, fontsize=20)
    plt.show()

def print_menu():
    print("Model Menu")
    print("=========================================================================")
    print("1. View KNN Model Score")
    print("2. View KNN Model Score with hyperparameter tuning ")
    print("3. View BEST KNN Model Prediction Score on Scraped Tweets")
    print("4. View AdaBoostClassifier Model Score")
    print("5. View AdaBoostClassifier Model Score with hyperparameter tuning ")
    print("6. View BEST AdaBoostClassifier Model Prediction Score on Scraped Tweets")
    print("7. View VotingClassifier Model Score with voting hard")
    print("8. View VotingClassifier Model Score with voting soft")
    print("9. View BEST VotingClassifier Model Prediction Score on Scraped Tweets")
    print("10. Train Data Details")
    print("11. Test Data Details")
    print("0. Exit")
    print("=========================================================================")

system_run = True
while(system_run):
    print_menu()
    selection = input("Enter (0-11)")
    print("\n")
    if selection == "1":
        runModel("knn.pkl")
    elif selection == "2":
        runModel("knn_tuning.pkl")
    elif selection == "3":
        runModel_To_Test(compareModel("knn.pkl", "knn_tuning.pkl"))
    elif selection == "4":
        runModel("adaBoost.pkl")
    elif selection == "5":
        runModel("adaBoost_tuning.pkl")
    elif selection == "6":
        runModel_To_Test(compareModel("adaBoost.pkl", "adaBoost_tuning.pkl"))
    elif selection == "7":
        runModel("vot_hard.pkl")
    elif selection == "8":
        runModel("vot_soft.pkl")
    elif selection == "9":
        runModel_To_Test(compareModel("vot_hard.pkl", "vot_soft.pkl"))
    elif selection == "10":
        showEmotionCountDiagram()
        showFrequencyCountDiagram()
        #showTestEmotionCountDiagram()
        #showTrainEmotionCountDiagram()
    elif selection == "11":
        readTweets.showEmotionCountDiagram()
        readTweets.showFrequencyCountDiagram()
    elif selection == "0":
        system_run = False
    else:
        print("Invalid Input, Please enter (0-11) only\n\n\n")
    print("\n")


