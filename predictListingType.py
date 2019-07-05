import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import sys

if(len(sys.argv)<2):
	print("please give test filenam")
filename = sys.argv[1]

model = joblib.load("xgboost_model.h5")

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

useless= ['Unnamed: 0', 'GUID', 'Name', 'Summary', 'Space', 'Description',
       'Experiences_Offered', 'Neighborhood_Overview', 'Notes', 'Transit',
       'Access', 'Interaction', 'House_Rules', 'Host_Name', 'Host_Since',
       'Host_Location', 'Host_About', 'Host_Neighbourhood','Neighbourhood_Cleansed',
       'Host_Listings_Count', 'Host_Total_Listings_Count', 'City',
       'State', 'Market', 'Smart_Location', 'Country_Code', 'Country',
	   'First_Review', 'Last_Review','Jurisdiction_Names','Geolocation',
	   'Calendar_last_Scraped','Calendar_Updated','Host_Response_Rate',
	   'Host_Acceptance_Rate','Neighbourhood_Group_Cleansed', 'Square_Feet',
	    'Weekly_Price', 'Monthly_Price', 'Security_Deposit', 'Cleaning_Fee',
		 'Has_Availability',  'License', 'Street','Zipcode','Features','Neighbourhood',
		 'Availability_30','Availability_60', 'Availability_90','Beds',"Host_Response_Time"]

non_categorical = ["Accommodates","Guests_Included","Maximum_Nights"]

categorical_list = {
		"Property_Type" : ['Apartment', 'House', 'Bed & Breakfast', 'Condominium', 'Townhouse', 'Loft', 'Other', 'Villa', 'Guesthouse'],
		"Cancellation_Policy" : ['strict', 'flexible', 'moderate', 'moderate_new', 'strict_new', 'flexible_new', 'super_strict_60'],
		"Room_Type": ["Entire home/apt","Private room"],
		"Bed_Type": ["Real Bed","Pull-out Sofa","Futon","Couch"]}

amenities_count=[]

def splitValues_To_Columns(data):
    data = data.str.split(",")
    for i in range(0,len(data)):
        temp = [x.replace(' ','') for x in data.iloc[i]]
        temp = [x.replace('-','') for x in temp]
        amenities_count.append(len(temp))
		
def groupdata(column,group_list,newdata):
	replace_str = "Other";
	newdata[column] = newdata[column].replace(np.nan, column + replace_str)
	counts_per_type = newdata[column].value_counts()
	newdata[column] = newdata[column].apply(lambda x : replace_str if x not in group_list else x)
 
def predictListingType(filename):
	#Read Data
	data = pd.read_csv(filename, delimiter=";")
	# Drop useless columns
	newdata= data.drop(useless,axis=1)
	#Get Amenities Count column
	# remove nan from amenities
	newdata["Amenities"] = newdata["Amenities"].replace(np.nan, "")
	splitValues_To_Columns(newdata["Amenities"])
	newdata["Amenities_Count"] = pd.Series(amenities_count)
	
	# convert wrongly interepted numerical variable as categorical
	for col in non_categorical:
		   newdata[col] = pd.to_numeric(newdata[col], errors='coerce')
   
   # fill NA numerical values by their mean
	for i in range(0, len(newdata.columns)):
		if (newdata.iloc[:,i].dtype  in numerics):
			 newdata.iloc[:,i] = newdata.iloc[:,i].replace(np.nan, newdata.iloc[:,i].mean())
	## Group the sparse data before one hot encoding:
	for i in range(0,len(categorical_list)):
		 columns = list(categorical_list.keys())
		 group_list = list(categorical_list.values())
		 groupdata(columns[i], group_list[i],newdata)
	## one hot encoding
	#print("Before DF size: {}".format(newdata.shape))
	newdata = pd.get_dummies(newdata.drop(["Amenities"], axis=1),drop_first=True)

	Y_Pred = model.predict(newdata)
	pd.Series(Y_Pred,name="Label").to_csv("result.csv", header=True)
	 
	 
predictListingType(filename)
	 
			 
	 
			 
	 
	
	
	
	
	