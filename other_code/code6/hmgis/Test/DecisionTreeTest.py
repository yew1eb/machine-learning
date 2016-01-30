#coding:utf-8

## 本例中我们展示了使用scikit的DT算法来进行决策树分类
## 决策树分类中，我们必须将本文数据的标量值转化为数值进行计算
## 即将ID3转换为C4.5

import numpy


class DecisionTreeDemo:
	def createDataset(self, infile):
		fr = open(infile)
		dataMat = []
		labels = []
		for item in fr.readlines():
			t = numpy.array(item.strip().split('\t'))
			a1 = -1
			if t[0] == "young":
				a1 = 0
			if t[0] == "pre":
				a1 = 1
			if t[0] == "presbyopic":
				a1 = 2

			a2 = -1
			if t[1] == "myope":
				a2 = 0
			if t[1] == "hyper":
				a2 = 1

			a3 = -1
			if t[2] == "yes":
				a3 = 0
			if t[2] == "no":
				a3 = 1

			a4 = -1
			if t[3] == "reduced":
				a4 = 0
			if t[3] == "normal":
				a4 = 1

			dataMat.append([a1, a2, a3, a4])

			a5 = -1
			if t[4] == "no lenses":
				a5 = 0
			if t[4] == "soft":
				a5 = 1
			if t[4] == "hard":
				a5 = 2
			labels.append(a5)

		return dataMat, labels

	def dtTest(self, dataMat, labels):
		## 决策树分类
		from sklearn import tree

		clf = tree.DecisionTreeClassifier()
		clf = clf.fit(dataMat, labels)
		print clf.predict([2, 0, 0, 1])
		#print clf

		## 将决策结果输出为PDF
		from sklearn.externals.six import StringIO
		import pydot

		dot_data = StringIO()
		tree.export_graphviz(clf, out_file=dot_data)
		graph = pydot.graph_from_dot_data(dot_data.getvalue())
		graph.write_pdf("data/dt/dt.pdf")
		print "输出PDF成功"



