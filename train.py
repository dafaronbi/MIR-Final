import numpy as np
import ds

#load data
train_data = ds.melParamData("train","data")
test_data = ds.melParamData("test","data")
validation_data = ds.melParamData("validation","data")

print(train_data[40])
print(test_data[40])
print(validation_data[400])