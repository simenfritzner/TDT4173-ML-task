from utilities import *
from constants import *

#Gustav sitt stygge arbeid 
X_estimated_a_edited = direct_rad_div_diffuse_rad(X_estimated_a)
X_estimated_b_edited = direct_rad_div_diffuse_rad(X_estimated_b)
X_estimated_c_edited = direct_rad_div_diffuse_rad(X_estimated_c)

X_test_a = direct_rad_div_diffuse_rad(X_test_a)
X_test_b = direct_rad_div_diffuse_rad(X_test_b)
X_test_c = direct_rad_div_diffuse_rad(X_test_c)


#WANT TO CHANGE THIS (NB with predictions form an estimator) 
#!!!!
#!!!!
#!!!!
#!!!!
#!!!!
y_b = drop_repeating_sequences(y_b.copy())
y_b = delete_ranges_of_zeros_and_interrupting_values(y_b.copy(),200,[0.8625])
y_b = delete_ranges_of_zeros_and_interrupting_values(y_b.copy(),25,[0.8625])
y_b = drop_long_sequences(y_b.copy(),25)
y_c = delete_ranges_of_zeros_and_interrupting_values(y_c.copy(),5,[19.6,9.8])

#but the rest is perfect:=0 

#Subset for March in X_observed_a
March_subset_1_X_observed_a = X_observed_a.iloc[29098-2:31977-1]  # Note that Python is 0-indexed and the ending index is exclusive
March_subset_2_X_observed_a = X_observed_a.iloc[64138-2:67017-1]
March_subset_3_X_observed_a = X_observed_a.iloc[99178-2:102057-1]

# Concatenate subsets for March
March_subset_X_observed_a = pd.concat([March_subset_1_X_observed_a, March_subset_2_X_observed_a, March_subset_3_X_observed_a])
March_subset_X_observed_a = direct_rad_div_diffuse_rad(March_subset_X_observed_a)

#Subset for March in X_observed_a for Mai
Mai_subset_1_X_observed_a = X_observed_a.iloc[31978-2:34953-1]  
Mai_subset_2_X_observed_a = X_observed_a.iloc[67018-2:69993-1]  
Mai_subset_3_X_observed_a = X_observed_a.iloc[102058-2:105033-1]  

# Concatenate subsets for Mai
Mai_subset_X_observed_a = pd.concat([Mai_subset_1_X_observed_a, Mai_subset_2_X_observed_a, Mai_subset_3_X_observed_a])
Mai_subset_X_observed_a = direct_rad_div_diffuse_rad(Mai_subset_X_observed_a)

#Subset for Juni in X_observed_a 
Juni_subset_1_X_observed_a = X_observed_a.iloc[2-2:2697-1]  
Juni_subset_2_X_observed_a = X_observed_a.iloc[34954-2:37833-1]  
Juni_subset_3_X_observed_a = X_observed_a.iloc[69994-2:72873-1] 
Juni_subset_4_X_observed_a = X_observed_a.iloc[105034-2:107923-1] 

# Concatenate subsets for Juni
Juni_subset_X_observed_a = pd.concat([Juni_subset_1_X_observed_a, Juni_subset_2_X_observed_a, Juni_subset_3_X_observed_a, Juni_subset_4_X_observed_a])
Juni_subset_X_observed_a = direct_rad_div_diffuse_rad(Juni_subset_X_observed_a)

#Subset for March in X_observed_a 
July_subset_1_X_observed_a = X_observed_a.iloc[2698-2:5673-1]  
July_subset_2_X_observed_a = X_observed_a.iloc[37834-2:40809-1]  
July_subset_3_X_observed_a = X_observed_a.iloc[72874-2:75844-1]  
July_subset_4_X_observed_a = X_observed_a.iloc[107914-2:110889-1]  

# Concatenate subsets for July
July_subset_X_observed_a = pd.concat([July_subset_1_X_observed_a, July_subset_2_X_observed_a, July_subset_3_X_observed_a, July_subset_4_X_observed_a])
July_subset_X_observed_a = direct_rad_div_diffuse_rad(July_subset_X_observed_a)


# Concatenate subsets for all dates 
subset_X_observed_a = pd.concat([March_subset_X_observed_a,Mai_subset_X_observed_a, Juni_subset_X_observed_a, July_subset_X_observed_a])


#Subset for March in X_observed_b
March_subset_1_X_observed_b = X_observed_b.iloc[8642-2:11521-1]  # Note that Python is 0-indexed and the ending index is exclusive
March_subset_2_X_observed_b = X_observed_b.iloc[43778-2:46657-1]
March_subset_3_X_observed_b = X_observed_b.iloc[78818-2:81697-1]
March_subset_4_X_observed_b = X_observed_b.iloc[113858-2:116737-1]

# Concatenate subsets for March
March_subset_X_observed_b = pd.concat([March_subset_1_X_observed_b, March_subset_2_X_observed_b, March_subset_3_X_observed_b,March_subset_4_X_observed_b ])
March_subset_X_observed_b = direct_rad_div_diffuse_rad(March_subset_X_observed_b)

#Subset for March in X_observed_b for Mai
Mai_subset_1_X_observed_b = X_observed_b.iloc[11522-2:14497-1]  
Mai_subset_2_X_observed_b = X_observed_b.iloc[46658-2:49633-1]  
Mai_subset_3_X_observed_b = X_observed_b.iloc[81698-2:84673-1]  
Mai_subset_4_X_observed_b = X_observed_b.iloc[116738-2:]   #to 116930

# Concatenate subsets for Mai
Mai_subset_X_observed_b = pd.concat([Mai_subset_1_X_observed_b, Mai_subset_2_X_observed_b, Mai_subset_3_X_observed_b, Mai_subset_4_X_observed_b])
Mai_subset_X_observed_b = direct_rad_div_diffuse_rad(Mai_subset_X_observed_b)

#Subset for Juni in X_observed_a 
Juni_subset_1_X_observed_b = X_observed_b.iloc[14498-2:17377-1]  
Juni_subset_2_X_observed_b = X_observed_b.iloc[49634-2:52513-1]  
Juni_subset_3_X_observed_b = X_observed_b.iloc[84674-2:87553-1] 

# Concatenate subsets for Juni
Juni_subset_X_observed_b = pd.concat([Juni_subset_1_X_observed_b, Juni_subset_2_X_observed_b, Juni_subset_3_X_observed_b])
Juni_subset_X_observed_b = direct_rad_div_diffuse_rad(Juni_subset_X_observed_b)

#Subset for March in X_observed_a 
July_subset_1_X_observed_b = X_observed_b.iloc[17378-2:20353-1]  
July_subset_2_X_observed_b = X_observed_b.iloc[52514-2:55489-1]  
July_subset_3_X_observed_b = X_observed_b.iloc[87554-2:90529-1]  

# Concatenate subsets for July
July_subset_X_observed_b = pd.concat([July_subset_1_X_observed_b, July_subset_2_X_observed_b, July_subset_3_X_observed_b])
July_subset_X_observed_b = direct_rad_div_diffuse_rad(July_subset_X_observed_b)


# Concatenate subsets for all dates 
subset_X_observed_b = pd.concat([March_subset_X_observed_b,Mai_subset_X_observed_b, Juni_subset_X_observed_b, July_subset_X_observed_b])

#Subset for March in X_observed_c
March_subset_1_X_observed_c = X_observed_c.iloc[8642-2:11521-1]  # Note that Python is 0-indexed and the ending index is exclusive
March_subset_2_X_observed_c = X_observed_c.iloc[43778-2:46657-1]
March_subset_3_X_observed_c = X_observed_c.iloc[78818-2:81697-1]
March_subset_4_X_observed_c = X_observed_c.iloc[113858-2:116737-1]

# Concatenate subsets for March
March_subset_X_observed_c = pd.concat([March_subset_1_X_observed_c, March_subset_2_X_observed_c, March_subset_3_X_observed_c, March_subset_4_X_observed_c])
March_subset_X_observed_c = direct_rad_div_diffuse_rad(March_subset_X_observed_c)


#Subset for March in X_observed_a for Mai
Mai_subset_1_X_observed_c = X_observed_c.iloc[11522-2:14497-1]  
Mai_subset_2_X_observed_c = X_observed_c.iloc[46658-2:49633-1]  
Mai_subset_3_X_observed_c = X_observed_c.iloc[81698-2:84673-1]  
Mai_subset_4_X_observed_c = X_observed_c.iloc[116738-2:] 

# Concatenate subsets for Mai
Mai_subset_X_observed_c = pd.concat([Mai_subset_1_X_observed_c, Mai_subset_2_X_observed_c, Mai_subset_3_X_observed_c, Mai_subset_4_X_observed_c])
Mai_subset_X_observed_c = direct_rad_div_diffuse_rad(Mai_subset_X_observed_c)


#Subset for Juni in X_observed_a 
Juni_subset_1_X_observed_c = X_observed_c.iloc[14498-2:17377-1]  
Juni_subset_2_X_observed_c = X_observed_c.iloc[49634-2:52513-1]  
Juni_subset_3_X_observed_c = X_observed_c.iloc[84674-2:87553-1] 

# Concatenate subsets for Juni
Juni_subset_X_observed_c = pd.concat([Juni_subset_1_X_observed_c, Juni_subset_2_X_observed_c, Juni_subset_3_X_observed_c])
Juni_subset_X_observed_c = direct_rad_div_diffuse_rad(Juni_subset_X_observed_c)


#Subset for March in X_observed_a 
July_subset_1_X_observed_c = X_observed_c.iloc[17378-2:20353-1]  
July_subset_2_X_observed_c = X_observed_c.iloc[52514-2:55489-1]  
July_subset_3_X_observed_c = X_observed_c.iloc[87554-2:90529-1]  

# Concatenate subsets for July
July_subset_X_observed_c = pd.concat([July_subset_1_X_observed_c, July_subset_2_X_observed_c, July_subset_3_X_observed_c])
July_subset_X_observed_c = direct_rad_div_diffuse_rad(July_subset_X_observed_c)


# Concatenate subsets for all dates 
subset_X_observed_c = pd.concat([March_subset_X_observed_c, Mai_subset_X_observed_c, Juni_subset_X_observed_c, July_subset_X_observed_c])


