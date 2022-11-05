#!/usr/bin/python2

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from ds_layer import NNCDS
import pickle

def loadData(datafile):
	features = np.loadtxt(datafile, delimiter=',', usecols=(0, 1, 2, 3))
	strLabels = np.loadtxt(datafile, delimiter=',', usecols=(4,), dtype=str)
	labelEnc = LabelEncoder()
	labelEnc.fit(strLabels)
	labels = labelEnc.transform(strLabels)
	print('Read a dataset from ' + datafile + ': ' + str(features.shape[0]) + ' data points, labels: ' + str(labelEnc.classes_))

	return (features, labels, labelEnc)


def getDecisionSpaceCrossSectionMap(
		classifier, fixDims, fixDimVals,
		freeDimsLimits, resolution, rejectionCost=None, newLabelCost=None):
	totalDims = len(fixDims) + len(freeDimsLimits)
	freeDims = list(range(totalDims))
	for dim in fixDims:
		freeDims.remove(dim)
	gridPts = [int(1+(end-start)/resolution) for start,end in freeDimsLimits]
	print('gridPts:' + str(gridPts))
	numPoints = np.product(gridPts)
	def linearIdxToGrid(linIdx, dim):
		curPoints = numPoints
		curIdx = linIdx
		for d in freeDims[:-1]:
			curPoints = curPoints/gridPts[freeDims.index(d)]
			coord = curIdx / curPoints
			curIdx = curIdx % curPoints
			if d == dim:
				return coord
		return curIdx
	grid = []
	for i in range(numPoints):
		curPos = [fixDimVals[fixDims.index(dim)] if dim in fixDims
				   else freeDimsLimits[freeDims.index(dim)][0] + resolution*linearIdxToGrid(i, dim) for dim in range(totalDims)]
		grid.append(curPos)
	gridVals = classifier.predict(grid, rejectionCost=rejectionCost, newLabelCost=newLabelCost)
	return gridVals.reshape(gridPts)

def discrete_cmap(N, base_cmap=None):
	""" Create an N-bin discrete colormap from the specified input map """

	# Note that if base_cmap is a string or None, you can simply do
	#    return plt.cm.get_cmap(base_cmap, N)
	# The following works for string, None, or a colormap instance:

	base = plt.cm.get_cmap(base_cmap)
	color_list = base(np.linspace(0, 1, N))
	cmap_name = base.name + str(N)
	return base.from_list(cmap_name, color_list, N)


def plotDecisionSpaceCrossSectionMap(classifier, fixDims, fixDimVals, freeDimsLimits, resolution, rejectionCost=None, newLabelCost=None):
	crossection = getDecisionSpaceCrossSectionMap(classifier, fixDims, fixDimVals, freeDimsLimits, resolution, rejectionCost=rejectionCost, newLabelCost=newLabelCost)
	totalDims = len(fixDims) + len(freeDimsLimits)
	freeDims = list(range(totalDims))
	for dim in fixDims:
		freeDims.remove(dim)
	if len(freeDims) != 2:
		raise ValueError('Only two-dimensional decision maps are supported, ' + str(len(freeDims)) + '-dimensional map implied by fixed dims')
	if len(freeDimsLimits) != 2:
		raise ValueError('Only two-dimensional decision maps are supported, ' + str(len(freeDims)) + '-dimensional map implied by limits')

	ranges = [ [ start + resolution*float(i) for i in range(int((end-start)/resolution) + 1) ] for start,end in freeDimsLimits ]
	# print('ranges len:' + str(len(ranges)))
	# print('ranges:' + str(ranges))
	x,y = np.meshgrid(ranges[0], ranges[1])
	# print('x:' + str(x.shape))
	# print('y:' + str(y.shape))
	# plt.plot(x, y, marker='1', color='red', linestyle='')
	# plt.show()
#	cs = plt.contour(x.T,y.T,crossection, colors='k', nchunk=0)
#	csf = plt.contourf(x.T,y.T,crossection, len(np.unique(crossection)), cmap=plt.cm.Paired)
	plt.pcolor(x.T, y.T, crossection, cmap=discrete_cmap(len(np.unique(crossection)), plt.cm.jet))
#	cb = plt.colorbar(ticks=np.unique(crossection), label='')
	plt.xlabel('petal length ' + str(freeDims[0]))
	plt.ylabel('petal width ' + str(freeDims[1]))

if __name__ == '__main__':
	irisDataFile = 'data/iris/iris.data'
	features, labels, labelEnc = loadData(irisDataFile)

	testVectors1 = [[6.2, 2.7, 5.5, 2.35], [6.4, 3.0, 4.71, 1.7]]  # Iris-virginica-2, Iris-versicolor-1

	eviclf = NNCDS()

	# eviclf.fit(features, labels, max_iterations=1000)
	# with open('model/eviclf.pickle', 'wb') as fw:
	# 	pickle.dump(eviclf, fw)

	with open('model/eviclf.pickle', 'rb') as fr:
		eviclfLoaded = pickle.load(fr)

	print('eviclfLoaded: Got labels ' + str(eviclfLoaded.predict(testVectors1)))

	# testLabelsAbs = eviclf.predict(testVectors1)
# #	print('Got labels ' + str(labelEnc.inverse_transform(testLabelsAbs)) + ' from ABS classifier (numericals ' + str(testLabelsAbs) + ')')
# 	print('eviclf: Got labels ' + str(testLabelsAbs))
#
# 	knncla = KNeighborsClassifier(n_neighbors=3)
# 	knncla.fit(features, labels)
# 	testLabelsKnn = knncla.predict(testVectors1)
# #	print('Got labels ' + str(labelEnc.inverse_transform(testLabelsKnn)) + ' from k-nn classifier (numericals ' + str(testLabelsKnn) + ')')
# 	print('KNN: Got labels ' + str(testLabelsKnn))
#
# 	plotDecisionSpaceCrossSectionMap(eviclf, [0, 1], [5.5, 3.0], [(-1, 8), (-1, 4)], 0.1, rejectionCost=0.5, newLabelCost=0.65)
# 	cb = plt.colorbar(ticks=[-2,-1,0,1,2])
# 	cb.set_ticklabels(['Novel', 'Reject', 'Iris Setosa', 'Iris Veriscolor', 'Iris Virginica'])
# 	plt.show()


