import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def getDict(gradePath, labelPath):
    gradeDf = pd.read_csv(gradePath, header=None)
    labelDf = pd.read_csv(labelPath, header=None)
    gradeDict = {}
    for i in range(len(gradeDf[0])):
        gradeDict[gradeDf[0][i]] = gradeDf[1][i]
    labelDict = {}
    for i in range(len(labelDf[0])):
        labelDict[labelDf[0][i]] = labelDf[1][i]
    return gradeDict, labelDict


def plot_scatter(gradeDict, labelDict):
    plt.figure(figsize=(10, 5))
    colors = ['green', 'orange', 'red']
    for key in gradeDict.keys():
        plt.scatter(gradeDict[key], 2, c=colors[labelDict[key]-1])
    plt.show()


def get_accurate(threshold, gradeDict, labelDict):
    acc = 0
    for key in gradeDict.keys():
        if gradeDict[key] > threshold[1] and labelDict[key] == 3:
            acc += 1
        elif gradeDict[key] < threshold[0] and labelDict[key] == 1:
            acc += 1
        elif threshold[0] < gradeDict[key] < threshold[1] and labelDict[key] == 2:
            acc += 1
    return round(acc / len(gradeDict.keys()), 4)


if __name__ == '__main__':
    gradeDict, labelDict = getDict("grades.csv", "label.csv")
    plot_scatter(gradeDict, labelDict)
    print(get_accurate([30, 35], gradeDict, labelDict))
