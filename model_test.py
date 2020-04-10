import pickle
from sklearn.linear_model import LinearRegression


model = pickle.load(open('model.pkl','rb'))
# print(model.predict([[80,1780000,6000,85]]))