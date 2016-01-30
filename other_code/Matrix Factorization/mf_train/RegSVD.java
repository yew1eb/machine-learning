package mf_train.mf_train;

import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.mahout.math.DenseMatrix;

public class RegSVD {
    
	private DenseMatrix F;
	private DenseMatrix G;
	private Random generator = new Random();
		
	/*Initialize
	 * 
	 * */
	public RegSVD(int rowOfF, int rowOfG, int k){
    	
		F = new DenseMatrix(rowOfF, k);
		G = new DenseMatrix(rowOfG, k);
	    	
	    //generate random number
	    for(int i=0 ; i<F.rowSize() ; ++i)
		  for(int j=0 ; j<F.columnSize() ; ++j)
			  F.set(i, j, generator.nextDouble());
	    
	    for(int i=0 ; i<G.rowSize() ; ++i)
			  for(int j=0 ; j<G.columnSize() ; ++j)
				  G.set(i, j, generator.nextDouble());
	}
	
	public void training(Map<Integer, Integer> userIdMap, Map<Integer, Integer> itemIdMap,
			              List<RatingData> trainingData, int maxIteration, double regParm,
			              double learningRate, double convergevalue)
	{
       int iteration=0;
       int k = F.columnSize();
       int u,v; //u:row for F, v: row for G
       double Buv, Ruv, Fus, Gvs; //Buv: current estimating rating, Ruv: current error for
       double[] MSE = new double[maxIteration];
       
       
       RatingData tempRating;
       
       System.out.println("start traing...........");
	   while(iteration < maxIteration){
		   
		   System.out.println("Iteration " + iteration);
		   Iterator<RatingData> iterator = trainingData.iterator();
		   
		   while(iterator.hasNext()){ //for all u,v in training set			  
			  
			  tempRating = iterator.next(); //get next training data 
			  
			  u = userIdMap.get(tempRating.getUserID()); //get to row # of user
			  v = itemIdMap.get(tempRating.getItemID()); // get the row # of item
			  
			  //System.out.println("u:" + u + "  v:" + v);
			  
			  Buv=F.viewRow(u).dot(G.viewRow(v)); //get current prediction
			  Ruv= tempRating.getRating()-Buv;  //get current error
			  
			  for(int s=0; s<k; ++s){
				  Fus= F.get(u,s)+ learningRate*(Ruv*G.get(v,s) - regParm*F.get(u,s) );
				  Gvs = G.get(v,s)+ learningRate*(Ruv*F.get(u,s) - regParm*G.get(v,s) );				  
				  F.set(u, s, Fus);
				  G.set(v, s, Gvs);				  
			  }//end for
			   
		   }//end inner while		   	   
		   
		   //calculate MSE for this iteration, if the MAE value is small enough
		   MSE[iteration] = calculateMSE(userIdMap,itemIdMap,trainingData);          
		   System.out.println("MSE in iteration " + iteration + " is:" + MSE[iteration]);
		   
		   //check convergence
		   if(iteration >=1){
			   
			   if( Math.abs(MSE[iteration]-MSE[iteration-1]) < convergevalue){				   
				   System.out.println("RMSE converged, stop running");
				   break;
			   }
		   }
		   
		   iteration++;	   
	   }//end while
	    
	}//end function	

	private double calculateMSE(Map<Integer, Integer> userIdMap, Map<Integer, Integer> itemIdMap,
	                            List<RatingData> trainingData)
	{	
		double SSE=0.0;
		int u,v; //u:row for F, v: row for G
		Iterator<RatingData> iterator = trainingData.iterator();
		RatingData tempRating;
		double  Buv, Ruv;
		
		 while(iterator.hasNext()){ //for all u,v in training set			  				
  		    
			 tempRating = iterator.next(); //get next training data 
		  
		     u = userIdMap.get(tempRating.getUserID()); //get to row # of user
		     v = itemIdMap.get(tempRating.getItemID()); // get the row # of item
		  
		     Buv=F.viewRow(u).dot(G.viewRow(v)); //get current prediction
		     Ruv= tempRating.getRating()-Buv;  //get current error
		     
		     SSE+=Math.pow(Ruv, 2.0);
		     //SSE+=Math.abs(Ruv);
		 }       
		 
		 SSE = Math.sqrt(SSE/trainingData.size());		 
   		 return SSE;
	}
	
	public double[] ratingPrediction(Map<Integer, Integer> userIdMap, Map<Integer, Integer> itemIdMap,
			List<RatingData> testingData)
	{
		double[] prediction = new double[testingData.size()];
		
		
		int u,v; //u:row for F, v: row for G
		RatingData tempRating;
	//	double Err=0,MAE=0;
		
		for(int i=0; i< testingData.size(); ++i){
			
			 tempRating = testingData.get(i); //get next training data 
		  
		     u = userIdMap.get(tempRating.getUserID()); //get to row # of user
		     v = itemIdMap.get(tempRating.getItemID()); // get the row # of item
		  
		     prediction[i]=F.viewRow(u).dot(G.viewRow(v)); //get current prediction
		  //   Err= tempRating.getRating()-prediction[i];  //get current error
		     
		    // if(Err > 4.0) System.out.println("Error > 4.0");
		     
		    // MAE+=Math.abs(Err);
		 }       
		// System.out.println("MAE="+ MAE/testingData.size());
		
		return prediction;
	}
	
	public static double MAE(double[] prediction, List<RatingData> testingData){
		
		double MAE=0.0, Err=0.0;
		RatingData tempRating;
		
		for(int i=0; i< testingData.size(); ++i){
			
			 tempRating = testingData.get(i); //get next training data 
		  
		     if(prediction[i] > 5.0) prediction[i]=5.0;
		     if(prediction[i] < 1.0) prediction[i]=1.0;
		     		    	 
		     Err= tempRating.getRating()-prediction[i];  //get current error		     		      
		     MAE+=Math.abs(Err);
		 }
		
		 //System.out.println("MAE="+ MAE/testingData.size());
				
		return MAE/testingData.size();
		
	}
	
public static double RMSE(double[] prediction, List<RatingData> testingData){
		
		double RMSE=0.0, Err=0.0;
		RatingData tempRating;
		
		for(int i=0; i< testingData.size(); ++i){
			
			 tempRating = testingData.get(i); //get next training data 
		  
		     if(prediction[i] > 5.0) prediction[i]=5.0;
		     if(prediction[i] < 1.0) prediction[i]=1.0;
		     		    	 
		     Err= tempRating.getRating()-prediction[i];  //get current error		     		      
		     RMSE+=Math.pow(Err,2.0);
		 }
		
		 //System.out.println("MAE="+ MAE/testingData.size());
				
		return Math.sqrt(RMSE/testingData.size());
		
	}
}

