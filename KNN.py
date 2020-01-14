import pandas as pd
from scipy.io import arff
from sklearn import preprocessing
import numpy as np

class KNN(object):
    def __init__(self,input_file):
        data = arff.loadarff(input_file)
        self.dataset = pd.DataFrame(data[0])
        self.K = int(np.sqrt(len(self.dataset)))
        if self.K % 2 == 0:
            self.K = self.K - 1

    def feature_normalization(self,feature_name):
        values = self.dataset[feature_name]
        min_val = min(values)
        max_val = max(values)
        self.dataset[feature_name] = [(x - min_val) / (max_val - min_val) for x in values]

    def normalize(self):
        cols = self.dataset[self.dataset.columns[:-1]]
        map(lambda x: self.feature_normalization(x),cols)

    def calculate_euclidian_distance(self,sample1, sample2):
        X = np.array(sample1)
        Y = np.array(sample2)
        m = len(sample1)
        e_distance_formula = ((np.sqrt((X - Y) ** 2)).sum()) / m
        return e_distance_formula

    def calc_distances(self,sample):
        distances = []
        for i in range(len(self.dataset)):
            next_sample = self.dataset.loc[i,:].tolist()[:-1]
            distances.append(self.calculate_euclidian_distance(sample,next_sample))
        return distances

    def classify_sample(self,sample):
        distances = self.calc_distances(sample)
        data = list(zip(distances, self.dataset['class']))
        df = pd.DataFrame(data, columns = ['Distance', 'class'])
        df.sort_values(['Distance'], inplace = True)
        df = df[:self.K]
        prediction = df['class'].value_counts().index[0]
        return prediction.decode("utf-8")
