package mf_train.mf_train;

import java.util.List;
import java.util.Map;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;

public class App 
{	
	/*file path*/
	final static String UserIDList = "./data/user_id.csv";
	final static String ItemIDList = "./data/item_id.csv";
	final static String trainData = "./data/train_data.csv";
	final static String testData = "./data/test_data.csv";
	
	public static void main( String[] args )
    {
		
		/*settings:
		 * K: number of features
		 * maxIter: max iterations
		 * learningRate: learning rate
		 * convergeValue: abs value between two iterations
		 * RegParm: regularized parameter
		 */
		int k=5;
		int maxIter=100;
		double learningRate=0.04;
		double convergevalue=0.0001;
		double regParm = 0.01;
		
		double RMSE=0.0;
		
		Map<Integer, Integer> userIdMap = DataImport.getMapFromCSV(UserIDList);
		Map<Integer, Integer> itemIdMap = DataImport.getMapFromCSV(ItemIDList);
		List<RatingData> trainingData = DataImport.getArrayFromCSV(trainData);
		List<RatingData> testingData = DataImport.getArrayFromCSV(testData);
		
		RegSVD svdModel = new RegSVD(userIdMap.size(), itemIdMap.size(),k);
		svdModel.training(userIdMap, itemIdMap, trainingData, maxIter, regParm, learningRate, convergevalue);
		
		double[] pred = svdModel.ratingPrediction(userIdMap, itemIdMap, testingData);
        
		 RMSE = RegSVD.RMSE(pred, testingData);
		System.out.println("RMSE=" + RMSE);
    }
}

