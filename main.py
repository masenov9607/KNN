from KNN import KNN

def main():
    input_file = "iris.arff"
    predictor = KNN(input_file)

    predictor.normalize()
    test_sample = [3,4,5,1.6]
    features = ["sepallength",  "sepalwidth",  "petallength",  "petalwidth"]
    print("sepal length : " + str(test_sample[0]))
    print("sepal widt : " + str(test_sample[1]))
    print("petal length : " + str(test_sample[2]))
    print("petal width : " + str(test_sample[3]))

    test_sample1 = [5.05666667, 3.26, 1.81166667, 0.38666667]
    test_sample2 =  [6.71794872, 3.02307692, 5.72564103, 2.08717949]
    test_sample3 =  [6.1,   2.83529412, 4.54509804, 1.4745098 ]

    a = predictor.classify_sample(test_sample1)
    print("\nPredicted flower is " + str(a))

    a = predictor.classify_sample(test_sample2)
    print("\nPredicted flower is " + str(a))

    a = predictor.classify_sample(test_sample3)
    print("\nPredicted flower is " + str(a))



if __name__ == "__main__":
    main()
