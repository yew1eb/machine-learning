package mf_train.mf_train;

import java.util.Random;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;

public class randomMatrixGenerator {

	static public DenseMatrix matrixgenerator(int nRow, int nCol){
		
    	DenseMatrix matrix = new DenseMatrix(nRow, nCol);
		Random generator = new Random();
				
		//generate random number
		for(int i=0 ; i<nRow ; ++i)
			for(int j=0 ; j<nCol ; ++j)
				matrix.set(i, j, generator.nextDouble());
		
	   return matrix;
	}
	
	/*
	int rowNum = 100;
	int colNum = 100;

	DenseMatrix testMatrix = new DenseMatrix(rowNum,colNum);
	DenseVector testVector = new DenseVector(10);
	
	testMatrix.set(0, 0, 10);
	testMatrix.set(1, 0, 9);
	testVector.assign(100.0);
	//DenseVector test2 = (DenseVector) testVector.normalize();
	//testVector = (DenseVector) testVector.normalize();
    */
}
