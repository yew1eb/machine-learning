package mf_train.mf_train;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

public class DataImport {
	
	public static Map<Integer, Integer> getMapFromCSV(String path){
		
		Map <Integer, Integer> map = new HashMap<Integer,Integer>();
		 int key;
		 int val;
		
		try{
		    File file = new File(path);
		    
		    //check if file exists.
		    if(file.isFile()){
		    	System.out.println("File: " + path + " is found.");
		    }
		    else{
		    	System.out.println("File: " + path + " is not found.");
		    }
			
		    BufferedReader bufRdr  = new BufferedReader(new FileReader(file));
		    String line = null;
		   
		    
		    //ignore the first line
		    bufRdr.readLine();
		    
		    
		    //read lines
            while((line = bufRdr.readLine()) != null){
		    	
            	StringTokenizer st = new StringTokenizer(line,",");
            	key = Integer.valueOf(st.nextToken());
            	val = Integer.valueOf(st.nextToken());
            	
            	map.put(key, val);
            	
		    }
		    
		    
		}
		catch(IOException e){
			System.out.println(e);
		}
		
		return(map);
		
	}

    
     public static List<RatingData> getArrayFromCSV(String path){
    	 
    	 List<RatingData> ratingList = new ArrayList<RatingData>();
    	 int userID;
    	 int itemID;
    	 int rating;
    	 
    	 try{
 		    File file = new File(path);
 		    
 		    //check if file exists.
 		    if(file.isFile()){
 		    	System.out.println("File: " + path + " is found.");
 		    }
 		    else{
 		    	System.out.println("File: " + path + " is not found.");
 		    }
 			
 		    BufferedReader bufRdr  = new BufferedReader(new FileReader(file));
 		    String line = null;
 		   
 		    
 		    //ignore the first line
 		    bufRdr.readLine();
 		    
 		    
 		    //read lines
             while((line = bufRdr.readLine()) != null){
 		    	
             	StringTokenizer st = new StringTokenizer(line,",");
             	userID = Integer.valueOf(st.nextToken());
             	itemID = Integer.valueOf(st.nextToken());
             	rating = Integer.valueOf(st.nextToken());
             	
             	ratingList.add(new RatingData(userID, itemID, rating));
             	
 		    }
 		    
 		    
 		}
 		catch(IOException e){
 			System.out.println(e);
 		}
    	 
    	 
    	 return ratingList;
     }
  }