package main

import (
	"fmt"

	base "github.com/sjwhitworth/golearn/base"
	evaluation "github.com/sjwhitworth/golearn/evaluation"
	knn "github.com/sjwhitworth/golearn/knn"
)

func main() {
	data, err := base.ParseCSVToInstances("iris_headers.csv", true)
	if err != nil {
		panic(err)
	}

	cls := knn.NewKnnClassifier("euclidean", 2)

	trainData, testData := base.InstancesTrainTestSplit(data, 0.8)
	cls.Fit(trainData)

	predictions := cls.Predict(testData)
	fmt.Println(predictions)

	confusionMat := evaluation.GetConfusionMatrix(testData, predictions)
	fmt.Println(evaluation.GetSummary(confusionMat))
}
