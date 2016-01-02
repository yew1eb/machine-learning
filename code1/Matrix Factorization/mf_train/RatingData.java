package mf_train.mf_train;

public class RatingData {
	
	private int userID;
	private int itemID;
	private int Rating;

	public RatingData(int uid, int iid, int rating){
		
		userID = uid;//generate F,G matrix
		itemID = iid;
		Rating = rating;	
	}
	
	public int getUserID(){
		return userID;
	}
	
	public int getItemID(){
		return itemID;
	}
	
	public int getRating(){
		return Rating;
	}
	
	public String toString(){
		String s ="User:" + userID + ", Item:" + itemID + " ,rating:" + Rating ;
		
		return s;
		
	}
}
