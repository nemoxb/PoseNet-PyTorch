import pickle
file = open('./places-googlenet.pickle', "rb")
weights = pickle.load(file, encoding="bytes")
print(weights.keys())