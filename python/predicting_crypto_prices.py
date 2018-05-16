'''
	This code implements a prediction for stocks/crypto prices using Support
	Vector Machine Regression. It is intended to serve as an example, thus, the
	code won't give a good prediction result.

Dependencies:
	- csv
	- numpy
	- scikit-learn
	- matplotlib

References:
	https://github.com/llSourcell/predicting_stock_prices
'''
import csv
import numpy as np
import datetime as dt
from sklearn.svm import SVR
import matplotlib.pyplot as plt



plt.switch_backend('TkAgg')



dates = []
prices = []

def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		# next(csvFileReader)	# skipping column names
		for row in csvFileReader:
			dates.append(row[0].split(' ')[0])
			prices.append(float(row[1]))
	return

def predict_price(dates, prices, x):
	# Limit the dataset to run faster (use the full dataset to get better results)

	dates = np.arange(len(dates))
	dates = np.reshape(dates,(len(dates), 1)) # converting to matrix of n X 1

	svr_lin = SVR(kernel= 'linear', C= 1e3)
	# svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
	svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # defining the support vector regression models
	svr_rbf.fit(dates, prices) # fitting the data points in the models
	svr_lin.fit(dates, prices)
	# svr_poly.fit(dates, prices)

	plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints
	plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel
	plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
	# plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') # plotting the line made by polynomial kernel
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()

	# return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]
	return svr_rbf.predict(x)[0], svr_lin.predict(x)[0]

bitcoin_prices = '../data/market/bitcoin_price.csv'
get_data(bitcoin_prices) # calling get_data method by passing the csv file to it
# print("Dates- ", dates)
# print("Prices- ", prices)

predicted_price = predict_price(dates, prices, 200)
