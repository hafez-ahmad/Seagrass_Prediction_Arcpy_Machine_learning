###############part 1################### 
#arcgis 
import arcpy
#  path
import os
#  data manipulation
import pandas as pd
import numpy as np
import numpy.ma as ma 

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
# machine learning
from sklearn.ensemble import RandomForestClassifier
# model evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
################part 2###################
# %%
# Set the workspace
arcpy.env.workspace = r"E:\1 Master's\MSU\course classes\Spring2023\GIS\project\data\data.gdb"
arcpy.env.overwriteOutput = True

# %%
# read data from geodatabase 
df = gpd.read_file(r"E:\1 Master's\MSU\course classes\Spring2023\GIS\project\data\EMU_Global_90m.shp")
# # dissO2 nitrate phosphate salinity silicate srtm30 temp
fields=["dissO2","nitrate","phosphate","salinity","silicate","srtm30","temp"]
# fill missing values with mean in fields
for field in fields:
    df[field].fillna(df[field].mean(), inplace=True)
#  write to geodatabase
df.to_file(r"E:\1 Master's\MSU\course classes\Spring2023\GIS\project\data\EMU_Global_90m_filled.shp")

# alternative 
# fill missing values with mean in EMU_Global_90m dataset
input_features = "variables"
# dissO2 nitrate phosphate salinity silicate srtm30 temp
fields=["dissO2","nitrate","phosphate","salinity","silicate","srtm30","temp","SHAPE@XY"]
out_table = "EMU_Global_90m_filled"
spatial_reference=arcpy.Describe(input_features).spatialReference
# convert feature class to pandas dataframe
df1 = arcpy.da.FeatureClassToNumPyArray(input_features,fields,spatial_reference=arcpy.Describe(input_features).spatialReference)
df1['nitrate']=np.where(np.isnan(df1['nitrate']), np.nanmean(df1['nitrate'], axis=0), df1['nitrate'])
df1['phosphate']=np.where(np.isnan(df1['phosphate']), np.nanmean(df1['phosphate'], axis=0), df1['phosphate'])
df1['salinity']=np.where(np.isnan(df1['salinity']), np.nanmean(df1['salinity'], axis=0), df1['salinity'])
df1['silicate']=np.where(np.isnan(df1['silicate']), np.nanmean(df1['silicate'], axis=0), df1['silicate'])
df1['srtm30']=np.where(np.isnan(df1['srtm30']), np.nanmean(df1['srtm30'], axis=0), df1['srtm30'])
df1['temp']=np.where(np.isnan(df1['temp']), np.nanmean(df1['temp'], axis=0), df1['temp'])
df1['dissO2']=np.where(np.isnan(df1['dissO2']), np.nanmean(df1['dissO2'], axis=0), df1['dissO2'])
outputDir = r"E:\1 Master's\MSU\course classes\Spring2023\GIS\project\data\data.gdb"
arcpy.env.overwriteOutput = True
arcpy.da.NumPyArrayToFeatureClass(df1, os.path.join(outputDir,out_table), ['SHAPE@XY'], spatial_reference)
## we need to move the file to the geodatabase
# %%
arcpy.env.overwriteOutput = True
# Create Random points constrain feature class: US_coastline_shallow
arcpy.management.CreateRandomPoints(arcpy.env.workspace, "Training", "US_coastline_shallow", "0 0 250 250", 10000, "0 DecimalDegrees", "POINT", 0)


# %%
fields=['dissO2', 'nitrate', 'phosphate', 'salinity', 'silicate', 'srtm30', 'temp']
# check license
if arcpy.CheckExtension("GeoStats") == "Available":
    arcpy.CheckOutExtension("GeoStats")
# for each interested values, I am going to do the empirical bayesian kriging and this will produce a raster for each variable
for field in fields:
    arcpy.ga.EmpiricalBayesianKriging("EMU_Global_90m_filled", field, None, field, 0.647000000001999, "NONE", 100, 1, 100, "NBRTYPE=StandardCircular RADIUS=15 ANGLE=0 NBR_MAX=15 NBR_MIN=10 SECTOR_TYPE=ONE_SECTOR", "PREDICTION", 0.5, "EXCEED", None, "POWER")
    print(f"{field} done")


# %%

# check license spatial analyst
if arcpy.CheckExtension("Spatial") == "Available":
    arcpy.CheckOutExtension("Spatial")
# extract values to points:ExtractMultiValuesToPoints
inpoint_features = "Training"
InRatserList=[f"{field}" for field in fields]
print(InRatserList)
arcpy.sa.ExtractMultiValuesToPoints(inpoint_features, fields)

# %%
# overwrite output
arcpy.env.overwriteOutput = True
input_features = "Training"
# add field : Present and Double
arcpy.AddField_management(input_features,"Present","DOUBLE")

# fill the field with 0
arcpy.management.CalculateField("Training", "Present", "0", "PYTHON3", '', "TEXT")

# select by location: Seagrass_USA then add 1 to the field
arcpy.management.SelectLayerByLocation("Training", "INTERSECT", "Seagrass_USA", None, "NEW_SELECTION", "NOT_INVERT")
arcpy.management.CalculateField(input_features, "Present", "1", "PYTHON3", '', "DOUBLE")

train_fc=input_features
Global_fc="EMU_Global_90m_filled"
fields=["dissO2","nitrate","phosphate","salinity","silicate","srtm30","temp",'Present',"SHAPE@XY"]
# predictor variables that are we going to use to predict the presence of seagrass
predictVars=["dissO2","nitrate","phosphate","salinity","silicate","srtm30","temp"]

# convert to structured numpy array
arr = arcpy.da.FeatureClassToNumPyArray(input_features,fields,spatial_reference=arcpy.Describe(input_features).spatialReference)

# flatten the structured numpy array
data = [tuple(row) for row in arr]
# convert to pandas dataframe
df = pd.DataFrame(data, columns=fields)
df.head()

################part 3###################

# %%
# split the data into training and test sets, frac=0.1 means 10% of the data will be used for testing
train_set=df.sample(frac=0.1,random_state=0)
# test
test_set=df.drop(train_set.index)

# %%
indicator_field ="Present" # the field that we want to predict
indicator,_=pd.factorize(train_set[indicator_field]) # convert the field to string to numeric
indicator

# %%
# print size of training and test sets
print(f"Training set size: {len(train_set)}")
print(f"Test set size: {len(test_set)}")

# %%
# random forest classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
# fit the model
rf.fit(train_set[predictVars], indicator)

# predict
seagreas_pred = rf.predict(test_set[predictVars])  

from sklearn.metrics import accuracy_score, precision_score, recall_score
# print accuracy, precision, recall, f1-score
print(f"Accuracy: {accuracy_score(test_set[indicator_field], seagreas_pred)}")
print(f"Precision: {precision_score(test_set[indicator_field], seagreas_pred)}")
print(f"Recall: {recall_score(test_set[indicator_field], seagreas_pred)}")
#  factorize the indicator field which means convert the string to number
indicator_USA,_=pd.factorize(df[indicator_field])
# train the model
model_rf=RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
# fit the model
model_rf.fit(df[predictVars], indicator_USA)

#read global data
arr = arcpy.da.FeatureClassToNumPyArray(Global_fc,['dissO2','nitrate','phosphate','salinity','silicate','srtm30','temp','SHAPE@XY'])
arr_ref=arcpy.Describe(Global_fc).spatialReference

global_train=pd.DataFrame(arr,columns=['dissO2','nitrate','phosphate','salinity','silicate','srtm30','temp'])
global_train.head()

# %%
seagrass_predicted=model_rf.predict(global_train[predictVars])
nameFC = 'Seagrass_Prediction'
outputDir = r"E:\1 Master's\MSU\course classes\Spring2023\GIS\project\data\data.gdb"
# find the index of the predicted seagrass is 1
grassExists = arr[["SHAPE@XY"]][global_train.index[np.where(seagrass_predicted==1)]]
# convert to feature class and save
arcpy.da.NumPyArrayToFeatureClass(grassExists, os.path.join(outputDir, nameFC), ['SHAPE@XY'], arr_ref)
################part 4###################
# check license spatial analyst
arcpy.CheckOutExtension("Spatial")
# kernel density, 0.2 is the cell size for the predicted feature class
out_raster = arcpy.sa.KernelDensity("Seagrass_Prediction", "NONE", 0.2, None, "SQUARE_MAP_UNITS", "DENSITIES", "PLANAR")
out_raster.save(r"E:\1 Master's\MSU\course classes\Spring2023\GIS\project\data\data.gdb\Predicted_seagrass")
del out_raster

print("done")



