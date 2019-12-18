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
			self.K += 1
    def featur_normalization(self,feature_name):
		values = self.dataset[feature_name]
		min_val = min(values)
		max_val = max(values)
		normalize_values = [(x - min_val) / (max_val - min_val) for x in values]

	def normalize_dataset(self):
		cols = [ "sepallength",  "sepalwidth",  "petallength",  "petalwidth"]
		min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
		np_scaled = preprocessing.normalize(self.dataset[cols])
		normalized_dataset = pd.DataFrame(np_scaled, columns = cols)
		self.dataset[cols] = normalized_dataset

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
		print(data)
		df = pd.DataFrame(data, columns = ['Distance', 'class'])
		df.sort_values(['Distance'], inplace = True)
		first_k_samples = df[0:self.K]
		print(first_k_samples)
		first_k_samples = first_k_samples.groupby(['class']).count()

		first_k_samples.sort_values(['Distance'],inplace = True)
		return first_k_samples
