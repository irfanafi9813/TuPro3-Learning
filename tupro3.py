# Library
import pandas as pd
import math
from random import seed
from random import randrange

# Membaca data latih/uji
def bacaFile():
    train = pd.read_excel('traintest.xlsx', sheet_name="train")
    train.drop('id', axis=1, inplace=True)
    test = pd.read_excel('traintest.xlsx', sheet_name="test")
    test.drop('id', axis=1, inplace=True)
    train = train.values.tolist()
    test = test.values.tolist()
    return [train, test]

# Pelatihan atau training model
def MeanAndStdDevForClass(mydata):
    info = {}
    dict = groupUnderClass(mydata)
    for classValue, instances in dict.items():
        info[classValue] = MeanAndStdDev(instances)
    return info

def groupUnderClass(mydata):
    global banyakDataTrain, banyakData0, banyakData1
    dict = {}
    for i in range(len(mydata)):
        if (mydata[i][-1] not in dict):
            dict[mydata[i][-1]] = []
        dict[mydata[i][-1]].append(mydata[i])
    banyakDataTrain = len(mydata)
    banyakData0 = len(dict[0])
    banyakData1 = len(dict[1])
    return dict

def MeanAndStdDev(mydata):
    info = [(mean(attribute), std_dev(attribute)) for attribute in zip(*mydata)]
    del info[-1]
    return info

def mean(numbers):
    return sum(numbers) / float(len(numbers))
 
def std_dev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)

def calculateClassProbabilities(info, test):
    probabilities = {}
    for classValue, classSummaries in info.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, std_dev = classSummaries[i]
            x = test[i]
            probabilities[classValue] *= calculateGaussianProbability(x, mean, std_dev)
        if classValue==1:
            probabilities[classValue] *= banyakData1/banyakDataTrain
        else:
            probabilities[classValue] *= banyakData0/banyakDataTrain
    return probabilities

def calculateGaussianProbability(x, mean, stdev):
    expo = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * expo

# Pengujian atau testing model
def getPredictions(info, test):
    predictions = []
    for i in range(len(test)):
        result = predict(info, test[i])
        predictions.append(result)
    return predictions

def predict(info, test):
    probabilities = calculateClassProbabilities(info, test)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

# Evaluasi model
# Membagi dataset menjadi k fold
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# evaluasi model menggunakan cross validation verification
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_rate(actual, predicted)
        scores.append(accuracy)
    return scores

def naive_bayes(train, test):
    summarize = MeanAndStdDevForClass(train)
    predictions = list()
    for row in test:
        output = predict(summarize, row)
        predictions.append(output)
    return(predictions)

def accuracy_rate(test, predictions):
    correct = 0
    for i in range(len(test)):
        if test[i] == predictions[i]:
            correct += 1
    return (correct / float(len(test))) * 100.0

# Menyimpan output ke file
def export(test, pred):
    for x in range(len(test)):
        test[x][3]=pred[x]
    df = pd.DataFrame(test, columns = ['x1','x2','x3','y'])
    df.to_excel('hasilNaiveBayes.xlsx', index=False)
    print("File berhasil di export")

def lain(dataset):
    data = groupUnderClass(dataset)
    data = pd.DataFrame.from_dict(data[1])
    data.columns = ['x1', 'x2', 'x3', 'y']
    print(data)

# Main program
mydata = bacaFile()
train_data = mydata[0]
test_data = mydata[1]

# print("Banyak data:", len(train_data))
# for n_folds in range(1, 51):
#     scores = evaluate_algorithm(train_data, naive_bayes, n_folds)
#     # print('Scores: %s' % scores)
#     print('Mean Accuracy-',n_folds,': %.3f%%' % (sum(scores)/float(len(scores))))

n_folds=5
scores = evaluate_algorithm(train_data, naive_bayes, n_folds)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

# predictions = getPredictions(info, train_data)
# accuracy = accuracy_rate1(train_data, predictions)
# print("Akurasi model:", accuracy)

info = MeanAndStdDevForClass(train_data)
predY = getPredictions(info, test_data)
# export(test_data, predY)

