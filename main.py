from KNN import KNN
import os
def main():
	solver = KNN("iris.arff")

	solver.normalize_dataset()
	test_sample = [5.1,3.5,1.4,0.2]
	#print(solver.calculate_euclidian_distance(test_sample,test_sample))
	solver.classify_sample(test_sample)



if __name__ == "__main__":
		main()
