#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pylab import plt
plt.style.use('seaborn')
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.stats as si
import scipy


# ### THESE ARE THE FUNCTIONS OF THE CODE:
# b) a) i)Annualize Vol from Hourly Data
# 
# b) a) ii)Annualize Vol from Daily Data
# 
# b) a) iii)Annualize Vol from Daily Data  with Sampling at Specific Time of Day 
# 
# 
# 
# 
# 
# 

# ### a) Read Time Series and Sets it to DataFrame

# ##### Load Up 10 Yr Bund Data

# In[177]:


data=pd.read_csv('BUND.csv') 
data.head(13)


# In[12]:


data['Price'].plot()


# In[16]:


data['Price'].var()


# ### b) a)

# In[21]:


type(data['Price'])


# In[26]:


#Takes the element in the 0 index, and every 4th element after that
list1 = data['Price'][::4]
list1.head()


# #### End of Day Sampling

# In[27]:


#Takes the element in the 10th index, which represents the 11th hour AND
# the daily close for the 1st day (considering a day with 11 trading hours),
# and every 11th element after that
list2 = data['Price'][10::11]
list2.head()


# In[ ]:





# In[33]:


# Annual Vol= sqrt(DAILY variance)*sqrt(252)
np.sqrt(list2.var())*np.sqrt(252)


# In[59]:


#Annual Vol=sqrt(HOULRY variance)*sqrt(2772)
#Assuming 252 trading days with 11 trading hours each
# 2772=252days*11hours per day
np.sqrt(data['Price'].var())*np.sqrt(2772)


# #### b) a) i)Annualize Vol from Hourly Data

# In[153]:


#Code to Annualize Vol from Hourly Data
def Hourly_Vol(frequency):
    Sample=data['Price'][frequency-1::frequency] #Creates New Sampling list based on frequency input
    Variance=Sample.var()    #Calculates Variance of New Sample
    Annual_Vol=np.sqrt(Variance)*np.sqrt(2772/frequency) # Annualizes Vol
    
    return Annual_Vol
    


# In[248]:


#Annual vol considering hourly data, sampling every 4 hours
Hourly_Vol(4)


# In[ ]:





# In[ ]:





# #### b) a) ii)Annualize Vol from Daily Data

# In[163]:


def Daily_Vol(frequency):
    Original_DAILY_Sample=data['Price'][10::11] #Grabs the Hourly Data and Converts it into Daily Data 
    NEW_Sample=Original_DAILY_Sample[frequency-1::frequency] #Creates New Sampling list based on frequency input
    Variance=NEW_Sample.var()    #Calculates Variance of New Sample
    Annual_Vol=np.sqrt(Variance)*np.sqrt(252/frequency) #Annualizes Vol
    
    return Annual_Vol


# In[250]:


#Annual vol considering daily data, sampling every 3 days
Daily_Vol(3)


# In[ ]:





# #### b) a) iii)Annualize Vol from Daily Data  with Sampling at Specific Time of Day 

# In[80]:


#inputs are: frequency of sampling(in days) AND sampling time of the day
#Assuming 11 trading hours a week
#sampling_time= the 'i'th hour of the trading day in which the data is sampled
def Daily_Vol_Specific_Hour(frequency,sampling_time):    
    Original_DAILY_Sample=data['Price'][sampling_time-1::11] #Grabs the Hourly Data and Converts it into Daily Data 
    NEW_Sample=Original_DAILY_Sample[frequency-1::frequency] #Creates New Sampling list based on frequency input
    Variance=NEW_Sample.var()    #Calculates Variance of New Sample
    Annual_Vol=np.sqrt(Variance)*np.sqrt(252/frequency) #Annualizes Vol
    
    return Annual_Vol


# In[88]:


#Annual vol considering DAILY data sampled on the 6th trading hour of the day
Daily_Vol_Specific_Hour(1,6)


# In[ ]:





# ### b) b) Space Increment Sampling Considering |Δ in bp| ≥ X bp

# In[244]:


def Space_Increment_Vol(increment_in_bp):
    Increment_Sample=[] #Create a List for the samples
    Increment_Sample.append(data['Price'][0]) #Add the first data point in our data

    Index_List=[1] #Create a list for the index of the samples

    #Loop to add the element in the list ONLY if its value is more than 5 bp away from last sample
    counter=0
    for i in data['Price']:
        counter+=1
        if abs(i-np.array(Increment_Sample[-1:]))>=(increment_in_bp/100):
            Increment_Sample.append(i)
            Index_List.append(counter)
        else:
            continue
    #CHECK THIS PART!!!!!!!!!!!!!!!!!!!  
    #Calculate the average distance (in hours) between each samples
    dt=np.array([x - Index_List[i - 1] for i, x in enumerate(Index_List)][1:]).mean()
    
    # Calculate Variance of the Sample
    np.array(Increment_Sample).var()
    
    #Calculate Annualized Vol based on average distance (in hours) between samples
    Annual_Vol=np.sqrt(np.array(Increment_Sample).var())*np.sqrt(2772/dt) #CHECK!
    
    return Annual_Vol


# In[251]:


#Variance sampling only moves ≥ 5bp from last Sampled Point
#Annualized vol based on conditional absolute size move (5bp) sampling
Space_Increment_Vol(5)


# In[ ]:





# #### c) a) i)Derived Time Series from Hourly Data

# In[159]:


def Hourly_Derived_Time_Series(frequency):
    Sample=data['Price'][frequency-1::frequency] #Creates New Sampling list based on frequency input
    
    return Sample


# In[252]:


#Derived hourly time series, sampling every 7th hour
Hourly_Derived_Time_Series(7).head()


# #### c) a) ii)Derived Time Series from Daily Data

# In[170]:


def Daily_Derived_Time_Series(frequency):
    Original_DAILY_Sample=data['Price'][10::11] #Grabs the Hourly Data and Converts it into Daily Data 
    NEW_Sample=Original_DAILY_Sample[frequency-1::frequency] #Creates New Sampling list based on frequency input
    
    return NEW_Sample


# In[175]:


#Derived daily time series, sampling every 5th day
Daily_Derived_Time_Series(5).head()


# #### c) a) iii)Derived Time Series from Daily Data with Sampling at Specific Time of Day

# In[178]:


def Daily_Derived_Time_Series_Specific_Hour(frequency,sampling_time):    
    Original_DAILY_Sample=data['Price'][sampling_time-1::11] #Grabs the Hourly Data and Converts it into Daily Data 
    NEW_Sample=Original_DAILY_Sample[frequency-1::frequency] #Creates New Sampling list based on frequency input
    
    return NEW_Sample


# In[253]:


#Derived daily time series, sampling every 5th day, on the 8th trading hour of the day
Daily_Derived_Time_Series_Specific_Hour(5,8).head()


# #### c) b) Derived Time Series from Space Increment Sampling Considering |Δ in bp| ≥ X bp

# In[240]:


def Space_Increment_Time_Series(increment_in_bp):
    Increment_Sample=[] #Create a List for the samples
    Increment_Sample.append(data['Price'][0]) #Add the first data point in our data

    Index_List=[1] #Create a list for the index of the samples

    #Loop to add the element in the list ONLY if its value is more than 5 bp away from last sample
    counter=0
    for i in data['Price']:
        counter+=1
        if abs(i-np.array(Increment_Sample[-1:]))>=(increment_in_bp/100):
            Increment_Sample.append(i)
            Index_List.append(counter)
        else:
            continue
    
    return pd.DataFrame(Increment_Sample)


# In[243]:


#Derived time series, sampling only moves ≥ 5bp from last Sampled Point
Space_Increment_Time_Series(5).head()


# In[ ]:





# ### d) Running Annualized Vol

# In[197]:


data['Price'].rolling(window=22).var().plot()


# In[ ]:


data['Price'].rolling(window=22).var()


# In[ ]:


np.sqrt(data['Price'].rolling(window=22).var())*np.sqrt()


# In[ ]:


data['Price'].rolling(22).var()


# 1- create new list with samples
# 
# 2- calculate the rolling variances based on those
# 
# 3- transform list of rolling variances to list of running ANNUAL vols

# In[213]:


#Rolling 1 Hour Variance
data['Price'].rolling(200).var().tail()


# In[217]:


#Rolling Annual Vol for 1 Hour with running window size=X
pd.DataFrame(np.sqrt(data['Price'].rolling(200).var())*np.sqrt(2772)).tail()


# In[ ]:


#Rolling Annual Vol for 1 Hour with running window size=X
pd.DataFrame(np.sqrt(data['Price'].rolling(200).var())*np.sqrt(2772/2)).tail()


# In[ ]:





# #### d)i) Running Annualized Vol with Hourly Samples

# In[218]:


def Running_Annual_Vol_with_Hourly_Samples(frequency,window):
    Sample=data['Price'][frequency-1::frequency] #Creates New Sampling list based on frequency input
    Running_Variance=Sample.rolling(window).var() #Calculates hourly running variance based on 'window size' input
    Running_Annual_Vol=np.sqrt(Running_Variance)*np.sqrt(2772/frequency)
    
    return pd.DataFrame(Running_Annual_Vol)


# In[221]:


Running_Annual_Vol_with_Hourly_Samples(1,200).plot()


# In[223]:


Running_Annual_Vol_with_Hourly_Samples(3,200).plot()


# In[ ]:





# #### d)ii) Running Annualized Vol with Daily Samples

# In[224]:


def Running_Annual_Vol_with_Daily_Samples(frequency,window):
    Original_DAILY_Sample=data['Price'][10::11] #Grabs the Hourly Data and Converts it into Daily Data 
    NEW_Sample=Original_DAILY_Sample[frequency-1::frequency] #Creates New Sampling list based on frequency input    
    Running_Variance=NEW_Sample.rolling(window).var() #Calculates daily running variance based on 'window size' input
    Running_Annual_Vol=np.sqrt(Running_Variance)*np.sqrt(252/frequency) 
    
    return pd.DataFrame(Running_Annual_Vol)


# In[232]:


Running_Annual_Vol_with_Daily_Samples(2,20).plot()


# In[ ]:





# #### d)iii) Running Annualized Vol with Daily Samples at Specific Time of Day

# In[233]:


def Running_Annual_Vol_with_Daily_Samples_on_Specific_Time_of_Day(frequency,sampling_time,window):
    Original_DAILY_Sample=data['Price'][sampling_time-1::11] #Grabs the Hourly Data and Converts it into Daily Data    
    NEW_Sample=Original_DAILY_Sample[frequency-1::frequency] #Creates New Sampling list based on frequency input       
    Running_Variance=NEW_Sample.rolling(window).var() #Calculates daily running variance based on 'window size' input
    Running_Annual_Vol=np.sqrt(Running_Variance)*np.sqrt(252/frequency) 
    
    return pd.DataFrame(Running_Annual_Vol)


# In[239]:


Running_Annual_Vol_with_Daily_Samples_on_Specific_Time_of_Day(2,11,20).plot()


# In[ ]:





# In[ ]:


#inputs are: frequency of sampling(in days) AND sampling time of the day
#Assuming 11 trading hours a week
#sampling_time= the 'i'th hour of the trading day in which the data is sampled
def Daily_Vol_Specific_Hour(frequency,sampling_time):    
    Original_DAILY_Sample=data['Price'][sampling_time-1::11] #Grabs the Hourly Data and Converts it into Daily Data 
    NEW_Sample=Original_DAILY_Sample[frequency-1::frequency] #Creates New Sampling list based on frequency input
    Variance=NEW_Sample.var()    #Calculates Variance of New Sample
    Annual_Vol=np.sqrt(Variance)*np.sqrt(252/frequency) #Annualizes Vol
    
    return Annual_Vol


# In[ ]:





# In[ ]:


def Daily_Vol(frequency):
    Original_DAILY_Sample=data['Price'][10::11] #Grabs the Hourly Data and Converts it into Daily Data 
    NEW_Sample=Original_DAILY_Sample[frequency-1::frequency] #Creates New Sampling list based on frequency input
    Variance=NEW_Sample.var()    #Calculates Variance of New Sample
    Annual_Vol=np.sqrt(Variance)*np.sqrt(252/frequency) #Annualizes Vol
    
    return Annual_Vol


# In[ ]:


def Running_Annual_Vol_with_Hourly_Samples(frequency,window):
    Sample=data['Price'][frequency-1::frequency] #Creates New Sampling list based on frequency input
    Running_Variance=Sample.rolling(window).var() #Calculates hourly running variance based on 'window size' input
    Running_Annual_Vol=np.sqrt(Running_Variance)*np.sqrt(2772/frequency)
    
    return pd.DataFrame(Running_Annual_Vol)


# In[ ]:





# In[ ]:


#could create 1 function ONLY for daily based on 1-frequency and 2-time of day sampling


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


def var_(values, frequency):
    avg = mean_(val, freq)
    dev = freq * (val - avg) ** 2
    return dev.sum() / (freq.sum() - 1)


# In[ ]:


def var_(val, freq):
    avg = mean_(val, freq)
    dev = freq * (val - avg) ** 2
    return dev.sum() / (freq.sum() - 1)

