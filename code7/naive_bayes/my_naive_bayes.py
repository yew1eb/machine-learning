


def prob(data, target, structure):
	count = structure
	for i, j in zip(target, data):
		try:
			count[i][j] += 1
		except:
			continue
	print count
	return [[math.log(j / sum(i)) for j in i] for i in count]


filename = 'train.csv'
splitRatio = .67
df = pd.read_csv(filename)
df = df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

meta = {
	'Pclass': [[0,0,0],[0,0,0]],
	'Sex': [[0,0],[0,0]],
	'Embarked': [[0,0,0],[0,0,0]]
}

pClassVar = {1:0, 2:1, 3:2}
df['Pclass'] = [pClassVar[i] for i in df['Pclass']]

sexVar = {'male':0, 'female':1}
df['Sex'] = [sexVar[i] for i in df['Sex']]

embarkedVar = {'Q':0, 'S':1, 'C':2}
df['Embarked'] = [embarkedVar[i] if i in embarkedVar else 3 for i in df['Embarked']]

length = len(df)
dividePoint = int(length * splitRatio)
trainingSet, testSet = df[:dividePoint], df[dividePoint:]
target = trainingSet['Survived']

meta['Pclass'] = prob(trainingSet['Pclass'], target, [[0,0,0],[0,0,0]])
meta['Sex'] = prob(trainingSet['Sex'], target, [[0,0],[0,0]])
meta['Embarked'] = prob(trainingSet['Embarked'], target, [[0,0,0],[0,0,0]])

# test = zip(testSet['Pclass'], testSet['Sex'], testSet['Embarked'], testSet['Survived'])

def predict(test, attributes):
	correct = 0
	for i in test:
		notSurvived = sum([calculate(i, attribute, 0) for attribute in attributes])
		survived    = sum([calculate(i, attribute, 1) for attribute in attributes])
		prediction = 1 if survived > notSurvived else 0
		print i
		if prediction == i[0]:
			correct += 1
	return correct / len(test)		

	# for i in test:
	# 	notSurvived = meta['Pclass'][0][i[0]] * meta['Sex'][0][i[1]] * meta['Embarked'][0][i[2]]
	# 	survived = meta['Pclass'][1][i[0]] * meta['Sex'][1][i[1]] * meta['Embarked'][1][i[2]]
	# 	prediction = 1 if survived > notSurvived else 0
	# 	if prediction == i[3]:
	# 		correct += 1
	# return correct / len(test)

def calculate(instance, attribute, targetVar):
	try:
		return meta[attribute][targetVar][instance[attribute]]
	except:
		return 0

# print testSet
# print predict(testSet, ['Pclass', 'Sex', 'Embarked'])

# count = defaultdict(int)
# for i in pClass:
# 	count[i] += 1
# print count

# for i in sex:
# 	count[i] += 1
# print count

# for i in embarked:
# 	count[i] += 1
# print count