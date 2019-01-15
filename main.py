import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

def solve(XX, eig, feature, dimension):     # Projection Onto the New Feature Space

	if (dimension == 1):
		matrix_w = np.hstack((eig[0][1].reshape(feature,1)))

	elif (dimension == 2):
		matrix_w = np.hstack((eig[0][1].reshape(feature,1), eig[1][1].reshape(feature,1)))

	elif(dimension == 3):
		matrix_w = np.hstack((eig[0][1].reshape(feature,1), eig[1][1].reshape(feature,1), eig[2][1].reshape(feature,1)))

	elif(dimension == 4):
		matrix_w = np.hstack((eig[0][1].reshape(feature,1), eig[1][1].reshape(feature,1), eig[2][1].reshape(feature,1), eig[3][1].reshape(feature,1)))

	elif(dimension == 5):
		matrix_w = np.hstack((eig[0][1].reshape(feature,1), eig[1][1].reshape(feature,1), eig[2][1].reshape(feature,1), eig[3][1].reshape(feature,1), eig[4][1].reshape(feature,1)))

	#print('Matrix W:\n', matrix_w)

	Y = XX.dot(matrix_w)

	return Y;


# *************************************************** Dataset  *******************************************************
dataset = pd.read_csv('train.txt')
dataset.drop(['id', 'date'], 1, inplace = True)

X = np.array(dataset.drop(['Occupancy'], 1))
y = np.array(dataset['Occupancy'])

# Normalization of data as different variables in data set may be having different units of measurement 
X_std = StandardScaler().fit_transform(X)

###############################################################################
###############################################################################
total_feature = 5                                                   ###########
selected_dimension = 3                                              ###########
###############################################################################
###############################################################################


# *******************************************  Reduce dimension from dataset ******************************************************************* #
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0] - 1)
#print('Covariance matrix \n%s' %cov_mat)


cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

#print('Eigenvectors \n%s' %eig_vecs)
#print('\nEigenvalues \n%s' %eig_vals)

for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

eig_pairs.sort()
eig_pairs.reverse()

#print('Eigenvalues in descending order:')
#for i in eig_pairs:
    #print(i[0])

X_new = solve(X_std, eig_pairs, total_feature, selected_dimension)

# ************************************************** Accuracy Check using SVM classifier *********************************************************** #

kf = StratifiedShuffleSplit(n_splits = 10, test_size = 0.1, random_state = 0)

# ************************************************** For original data *********************************************************** #
sum = 0.0
for train_index, test_index in kf.split(X, y):
	#print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	model = svm.SVC(gamma = 'auto')
	model.fit(X_train, y_train)
	accuracy = model.score(X_test, y_test)
	sum += accuracy
	#print(accuracy)
accuracy = sum/(kf.get_n_splits(X, y)*1.0);
print('Accuracy of Model = %s' %accuracy)

# ************************************************** For reduced dimensional data *********************************************************** #
sum = 0.0
for train_index, test_index in kf.split(X_new, y):
	#print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = X_new[train_index], X_new[test_index]
	y_train, y_test = y[train_index], y[test_index]
	model = svm.SVC(gamma = 'auto')
	model.fit(X_train, y_train)
	accuracy = model.score(X_test, y_test)
	sum += accuracy
	#print(accuracy)
accuracy = sum/(kf.get_n_splits(X, y)*1.0);
print('After applying PCA , Accuracy of Model = %s' %accuracy)

# ************************************************************ END ********************************************************************************* #
