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
# b) b) Annualize Vol with Space Increment Sampling Considering |Δ in bp| ≥ X bp
# 
# c) a) i)Derived Time Series from Hourly Data
# 
# c) a) ii)Derived Time Series from Daily Data
# 
# c) a) iii)Derived Time Series from Daily Data with Sampling at Specific Time of Day
# 
# c) b) Derived Time Series from Space Increment Sampling Considering |Δ in bp| ≥ X bp
# 
# d)a)i) Running Annualized Vol with Hourly Samples
# 
# d)a)ii) Running Annualized Vol with Daily Samples
# 
# d)a)iii) Running Annualized Vol with Daily Samples at Specific Time of Day
# 
# d)b) Running Annualized Vol with Daily Samples at Specific Time of Day
# 
# e)a)i) Plot of Running Annual Vol and Derived Time Series using Hourly Data
# 
# e)a)ii) Plot of Running Annual Vol and Derived Time Series using Daily Data
# 
# e)a)iii) Plot of Running Annual Vol and Derived Time Series using Daily Data(Sampled at Specific Time of Day)
# 
# e)b) Plot of Running Annual Vol and Derived Time Series from Space Increment Sampling Considering |Δ in bp| ≥ X bp

# ### a) Read Time Series and Sets it to DataFrame

# ##### Load Up 10 Yr Bund Data

# In[471]:


DATES=pd.read_csv('BUND_YIELD.csv')['Date'] #Create dataframe with Dates
DATES=DATES[-len(DATES)+1:] #Eliminate very first date data point because it doesnt have a return to it
DATES.head()


# In[783]:


#Load Up Price Data
data=pd.read_csv('BUND_YIELD.csv',index_col=0) 
data.head(13)


# In[784]:


#Plot Price Data
ax=data['Price'].plot(rot=90)
plt.xlabel('Date')
plt.ylabel('Yield in %') 
plt.title('10 Year Bund Yield')


# In[785]:


#Calculate log returns
data['Returns'] = np.log(data.Price) - np.log(data.Price.shift(1))
data.head()


# In[786]:


#Remove NaN values from data set
data=data.dropna(subset=['Returns'])
data.head()


# In[787]:


#Calculate Variance of the entire sample
data['Returns'].var(skipna=True)


# ### b) a)

# Annual Vol (Using Daily Data)= sqrt(DAILY variance)*sqrt(# Trading Days in a Year)
# 
# Annual Vol (Using Hourly Data)=sqrt(HOURLY variance)*sqrt(# Trading Hours in a Year)
# 
# Number of Trading Days in a Year=252
# 
# Number of Trading Hours in a Year=252 Days * Trading Hours in a Days= Total Trading Hours in a Year
# 

# #### b) a) i)Annualize Vol from Hourly Data

# In[788]:


#Code to Annualize Vol from Hourly Data
def Hourly_Vol(frequency,trading_hours_per_day):
    Trading_Hours_in_Trading_Year=252*trading_hours_per_day #Calculates Trading Hours in a year
    Sample=data['Price'][frequency-1::frequency] #Creates New Sampling list based on frequency input
    Returns=np.log(Sample) - np.log(Sample.shift(1)) #Calculates Returns on New Sample    
    Variance=Returns.var(skipna=True)    #Calculates Variance of New Sample
    Annual_Vol=np.sqrt(Variance)*np.sqrt(Trading_Hours_in_Trading_Year/frequency) # Annualizes Vol
    
    return Annual_Vol   


# In[789]:


#Annual vol considering hourly data, sampling every 1 hours.
#Considers 1 trading hour in a trading day.
Hourly_Vol(1,1)


# In[790]:


#Annual vol considering hourly data, sampling every 4 hours.
#Considers 11 trading hours in a trading day.
Hourly_Vol(4,11)


# #### b) a) ii)Annualize Vol from Daily Data

# In[316]:


def Daily_Vol(frequency,trading_hours_per_day):
    Original_DAILY_Sample=data['Price'][trading_hours_per_day-1::trading_hours_per_day] #Grabs the Hourly Data and Converts it into Daily Data. According to # trading hours per day 
    NEW_Sample=Original_DAILY_Sample[frequency-1::frequency] #Creates New Sampling list based on frequency input
    Returns=np.log(NEW_Sample) - np.log(NEW_Sample.shift(1)) #Calculates Returns on New Sample    
    Variance=Returns.var(skipna=True)    #Calculates Variance of New Sample

    Annual_Vol=np.sqrt(Variance)*np.sqrt(252/frequency) #Annualizes Vol
    
    return Annual_Vol


# In[317]:


#Annual vol considering daily data, sampling every 3 days.
#Considers 11 trading hours in a trading day
Daily_Vol(3,11)


# In[318]:


#Annual vol considering daily data, sampling every 2 days.
#Considers 1 trading hour in a trading day
Daily_Vol(2,1)


# #### b) a) iii)Annualize Vol from Daily Data  with Sampling at Specific Time of Day 

# In[307]:


#inputs are: frequency of sampling(in days), sampling time of the day (in hours) and trading hours per day
#sampling_time= the 'i'th hour of the trading day in which the data is sampled
def Daily_Vol_Specific_Hour(frequency,sampling_time,trading_hours_per_day):    
    Original_DAILY_Sample=data['Price'][sampling_time-1::trading_hours_per_day] #Grabs the Hourly Data and Converts it into Daily Data based on Sampling Time of Day AND Trading Hours per Day 
    NEW_Sample=Original_DAILY_Sample[frequency-1::frequency] #Creates New Sampling list based on sampling frequency input
    Returns=np.log(NEW_Sample) - np.log(NEW_Sample.shift(1)) #Calculates Returns on New Sample    
    Variance=Returns.var(skipna=True)    #Calculates Variance of New Sample    
    
    Annual_Vol=np.sqrt(Variance)*np.sqrt(252/frequency) #Annualizes Vol
    
    return Annual_Vol


# In[308]:


#Annual vol considering DAILY data sampled on the 6th trading hour of the day.
#Considers 8 trading hours in a trading day
Daily_Vol_Specific_Hour(1,6,8)


# In[294]:


#Annual vol considering data sampled every other day, on the 7th trading hour of the day
#Considers 11 trading hours in a trading day
Daily_Vol_Specific_Hour(2,7,11)


# ### b) b) Space Increment Sampling Considering |Δ in bp| ≥ X bp

# In[547]:


def Space_Increment_Vol(increment_in_bp,trading_hours_per_day):
    Trading_Hours_in_Trading_Year=252*trading_hours_per_day #Calculates Trading Hours in a year
    Increment_Sample=[] #Create a List for the samples
    Increment_Sample.append(data['Price'][0]) #Add the first data point in our data

    Index_List=[1] #Create a list for the index of the samples

    #Loop to add the element in the list ONLY if its value is more than X bp away from last sample
    counter=0 #counter to keep track in which hours did we sample
    for i in data['Price']:
        counter+=1
        if abs(i-np.array(Increment_Sample[-1:]))>=(increment_in_bp/100):
            Increment_Sample.append(i)
            Index_List.append(counter)
        else:
            continue
 
    #Calculate the average distance (in hours) between each samples
    dt=np.array([x - Index_List[i - 1] for i, x in enumerate(Index_List)][1:]).mean()
    
    #Convert to a DataFrame
    Increment_Sample=pd.DataFrame(Increment_Sample)
    
    #Calculate Returns on NEW sample
    Returns=np.log(Increment_Sample) - np.log(Increment_Sample.shift(1))
    
    # Calculate Variance
    Variance=Returns.var(skipna=True)
    
    #Calculate Annualized Vol based on average distance (in hours) between samples (dt) and the trading hours in a year
    Annual_Vol=np.sqrt(Variance)*np.sqrt(Trading_Hours_in_Trading_Year/dt) #
    
    return Annual_Vol


# In[548]:


#Variance sampling only moves ≥ 5bp from last Sampled Point
#Annualized vol based on conditional absolute size move (5bp) sampling
#Considering 8 trading hours per day
Space_Increment_Vol(5,8)


# In[ ]:





# #### c) a) i)Derived Time Series from Hourly Data

# In[63]:


def Hourly_Derived_Time_Series(frequency):
    Sample=data['Price'][frequency-1::frequency] #Creates New Sampling list based on frequency input
    
    return pd.DataFrame(Sample)


# In[437]:


#Derived hourly time series, sampling every 7th hour
Hourly_Derived_Time_Series(7).head()


# #### c) a) ii)Derived Time Series from Daily Data

# In[321]:


def Daily_Derived_Time_Series(frequency,trading_hours_per_day):
    Original_DAILY_Sample=data['Price'][trading_hours_per_day-1::trading_hours_per_day] #Grabs the Hourly Data and Converts it into Daily Data. According to # trading hours per day 
    NEW_Sample=Original_DAILY_Sample[frequency-1::frequency] #Creates New Sampling list based on frequency input
    
    return pd.DataFrame(NEW_Sample)


# In[326]:


#Derived daily time series, sampling every 3rd day
#Considers 8 trading hours per day
Daily_Derived_Time_Series(1,8).head()


# #### c) a) iii)Derived Time Series from Daily Data with Sampling at Specific Time of Day

# In[328]:


def Daily_Derived_Time_Series_Specific_Hour(frequency,sampling_time,trading_hours_per_day):    
    Original_DAILY_Sample=data['Price'][sampling_time-1::trading_hours_per_day] #Grabs the Hourly Data and Converts it into Daily Data based on Trading Hours per day and Sampling time of day 
    NEW_Sample=Original_DAILY_Sample[frequency-1::frequency] #Creates New Sampling list based on frequency input
    
    return pd.DataFrame(NEW_Sample)


# In[329]:


#Derived daily time series, sampling every 5th day, on the 8th trading hour of the day
#Considers 10 trading hours per day
Daily_Derived_Time_Series_Specific_Hour(5,8,10).head()


# #### c) b) Derived Time Series from Space Increment Sampling Considering |Δ in bp| ≥ X bp

# In[558]:


def Space_Increment_Time_Series(increment_in_bp):
    Increment_Sample=[] #Create a List for the samples
    Dates_List=[] #Create list for Dates
    
    Increment_Sample.append(data['Price'][0]) #Add the first data point in our data
    Dates_List.append(DATES[0:1]) #Add the first

    Index_List=[1] #Create a list for the index of the samples

    #Loop to add the element in the list ONLY if its value is more than X bp away from last sample
    counter=0 #counter to keep track in which hours did we sample
    for i in data['Price']:
        counter+=1
        if abs(i-np.array(Increment_Sample[-1:]))>=(increment_in_bp/100):
            Increment_Sample.append(i)
            Index_List.append(counter)
            Dates_List.append(DATES[counter])
            
        else:
            continue
    
    Final_DF=pd.DataFrame(Increment_Sample,columns=['Price']) #Create dataframe with sampled prices
    Final_DF['Dates']=Dates_List   #Add sampled dates to the dataframe
    Final_DF.set_index('Dates',inplace=True, drop=True) #Set Dates as the index
    
    return Final_DF


# In[563]:


#Derived time series, sampling only moves ≥ 5bp from last Sampled Point
Space_Increment_Time_Series(5).head()


# ### d) Running Annualized Vol

# In[332]:


#Rolling 1 Hour Variance with window size of 200 hours(data points)
data['Returns'].rolling(200).var().plot(rot=90)
plt.xlabel('Date')
plt.ylabel('Hourly Variance') 
plt.title('Rolling 1 Hour Variance with window size of 200 hours')


# In[ ]:





# In[ ]:





# In[333]:


#Rolling Annual Vol for 1 Hour with running window size=200
pd.DataFrame(np.sqrt(data['Returns'].rolling(200).var())*np.sqrt(2772)).plot(rot=90)
plt.xlabel('Date')
plt.ylabel('Annual Volatility (%)') 
plt.title('Rolling Annual Vol for 1 Hour Data with window size=200')


# #### d)a)i) Running Annualized Vol with Hourly Samples

# In[564]:


def Running_Annual_Vol_with_Hourly_Samples(frequency,window,trading_hours_per_day):
    Trading_Hours_in_Trading_Year=252*trading_hours_per_day #Calculates Trading Hours in a year    
    Sample=data['Price'][frequency-1::frequency] #Creates New Sampling list based on frequency input
    Returns=np.log(Sample) - np.log(Sample.shift(1)) #Calculates Returns on New Sample    
    
    Running_Variance=Returns.rolling(window).var() #Calculates hourly running variance based on 'window size' input
    Running_Annual_Vol=np.sqrt(Running_Variance)*np.sqrt(Trading_Hours_in_Trading_Year/frequency)
    
    return pd.DataFrame(Running_Annual_Vol)


# In[565]:


#Running Annual Vol, Sampling every Hour and with window size of 200 hours
#Considers 8 trading hours in a day
Running_Annual_Vol_with_Hourly_Samples(1,200,8).plot(rot=90)


# In[566]:


#Running Annual Vol, Sampling every 3 Hours and with window size of 200 
#Considers 11 trading hours in a day
Running_Annual_Vol_with_Hourly_Samples(3,200,11).plot()


# #### d)a)ii) Running Annualized Vol with Daily Samples

# In[567]:


def Running_Annual_Vol_with_Daily_Samples(frequency,window,trading_hours_per_day):
    Original_DAILY_Sample=data['Price'][trading_hours_per_day-1::trading_hours_per_day] #Grabs the Hourly Data and Converts it into Daily Data. According to # trading hours per day 
    NEW_Sample=Original_DAILY_Sample[frequency-1::frequency] #Creates New Sampling list based on frequency input    
    Returns=np.log(NEW_Sample) - np.log(NEW_Sample.shift(1)) #Calculates Returns on New Sample    
    Running_Variance=Returns.rolling(window).var() #Calculates daily running variance based on 'window size' input
    Running_Annual_Vol=np.sqrt(Running_Variance)*np.sqrt(252/frequency)  
    
    return pd.DataFrame(Running_Annual_Vol)


# In[568]:


#Running Annual Vol, Sampling every 2 Days and with window size of 200 days 
#Considers 8 trading hours in a day
Running_Annual_Vol_with_Daily_Samples(2,20,8).plot(rot=90)


# In[ ]:





# #### d)a)iii) Running Annualized Vol with Daily Samples at Specific Time of Day

# In[672]:


def Running_Annual_Vol_with_Daily_Samples_on_Specific_Time_of_Day(frequency,sampling_time,window,trading_hours_per_day):
    Original_DAILY_Sample=data['Price'][sampling_time-1::trading_hours_per_day] #Grabs the Hourly Data and Converts it into Daily Data based on Sampling Time of Day AND Trading Hours per Day 
    NEW_Sample=Original_DAILY_Sample[frequency-1::frequency] #Creates New Sampling list based on sampling frequency input
    Returns=np.log(NEW_Sample) - np.log(NEW_Sample.shift(1)) #Calculates Returns on New Sample    
    Running_Variance=Returns.rolling(window).var() #Calculates daily running variance based on 'window size' input
    Running_Annual_Vol=np.sqrt(Running_Variance)*np.sqrt(252/frequency) 
    
    return pd.DataFrame(Running_Annual_Vol)


# In[673]:


#Running Annual Vol Sampling every 2 days on the 11th trading hour of the day, with window size=20
#Considers 8 trading hours in a day
Running_Annual_Vol_with_Daily_Samples_on_Specific_Time_of_Day(2,11,20,8).plot(rot=90)


# In[ ]:





# #### d)b) Running Annualized Vol with Daily Samples at Specific Time of Day

# In[644]:


def Running_Vol_with_Space_Increment_Sampling(increment_in_bp,window,trading_hours_per_day):
    Trading_Hours_in_Trading_Year=252*trading_hours_per_day #Calculates Trading Hours in a year
    Increment_Sample=[] #Create a List for the samples
    Dates_List=[] #Create list for Dates    
    
    Increment_Sample.append(data['Price'][0]) #Add the first data point in our data
    Dates_List.append(DATES[0:1]) #Add the first Date


    Index_List=[1] #Create a list for the index of the samples

    #Loop to add the element in the list ONLY if its value is more than X bp away from last sample
    counter=0 #counter to keep track in which hours did we sample
    for i in data['Price']:
        counter+=1
        if abs(i-np.array(Increment_Sample[-1:]))>=(increment_in_bp/100):
            Increment_Sample.append(i)
            Index_List.append(counter)
            Dates_List.append(DATES[counter])

        else:
            continue
    
    #Convert to a DataFrame
    Increment_Sample=pd.DataFrame(Increment_Sample)
    
    #Calculate Returns on NEW sample
    Returns=np.log(Increment_Sample) - np.log(Increment_Sample.shift(1))
    
    #Calculate Running average distance(in hours) between samples
    dt=pd.DataFrame([x - Index_List[i - 1] for i, x in enumerate(Index_List)][1:]).rolling(window).mean()
    
    # Calculate Running Variance
    Running_Variance=Returns.rolling(window).var() #Calculates hourly running variance based on 'window size' input
    
    #Calculate Annualized Vol based on average distance (in hours) between samples (dt) and the trading hours in a year
    Annual_Vol=np.sqrt(Running_Variance)*np.sqrt(Trading_Hours_in_Trading_Year/dt) 
    
    Final_DF=pd.DataFrame(Annual_Vol)#Create dataframe with Running Annual Vol
    Final_DF['Dates']=Dates_List   #Add sampled dates to the dataframe
    Final_DF.set_index('Dates',inplace=True, drop=True) #Set Dates as the index
    Final_DF.columns = ['Running Annual Vol']  #Assign column name
    
    return Final_DF


# In[647]:


#Running Annualized vol sampling only when theres a move larger than 5bp from previous sampled point
#Considers window size of 50 hours and 8 trading hours in a day
Running_Vol_with_Space_Increment_Sampling(5,50,8).plot(rot=90)


# ### e) Plotting c and d on the same graph

# #### e)i) HOURLY

# In[781]:


def CHART_Running_Annual_Vol_with_Hourly_Samples(frequency,window,trading_hours_per_day):
    Trading_Hours_in_Trading_Year=252*trading_hours_per_day #Calculates Trading Hours in a year    
    Sample=data['Price'][frequency-1::frequency] #Creates New Sampling list based on frequency input
    Returns=np.log(Sample) - np.log(Sample.shift(1)) #Calculates Returns on New Sample    
    
    Running_Variance=Returns.rolling(window).var() #Calculates hourly running variance based on 'window size' input
    Running_Annual_Vol=np.sqrt(Running_Variance)*np.sqrt(Trading_Hours_in_Trading_Year/frequency)
    
    #Place Running Vols and Time Series in DataFrame
    DF=pd.DataFrame(Sample)
    DF['Running_Vol']=Running_Annual_Vol
    
    #Create Plot
    DF.Price.plot()
    plt.legend()
    plt.ylabel('Yield (%)')
    DF.Running_Vol.plot(secondary_y=True, style='g',rot=90)
    plt.xlabel('Date')
    plt.ylabel('Running Vol') 
    plt.title('10 Year Bund Yield vs Annualized Running Vol (Window Size=200)')
    plt.legend(bbox_to_anchor=(0.8, 1))
    plt.text(0.8, 5.4, "Frequency={}. Window Size={}. Trading Hours per Day={}".format(frequency, window,trading_hours_per_day))

    
    return plt


# In[782]:


#Plots Running Annual Vol vs Price with two vertical axis
#Considers sampling every hour, window size of 200 and 8 trading hours in a day
CHART_Running_Annual_Vol_with_Hourly_Samples(1,200,8);


# In[ ]:





# #### e)ii) DAILY

# In[773]:


def CHART_Running_Annual_Vol_with_Daily_Samples(frequency,window,trading_hours_per_day):
    Original_DAILY_Sample=data['Price'][trading_hours_per_day-1::trading_hours_per_day] #Grabs the Hourly Data and Converts it into Daily Data. According to # trading hours per day 
    NEW_Sample=Original_DAILY_Sample[frequency-1::frequency] #Creates New Sampling list based on frequency input    
    Returns=np.log(NEW_Sample) - np.log(NEW_Sample.shift(1)) #Calculates Returns on New Sample    
    Running_Variance=Returns.rolling(window).var() #Calculates daily running variance based on 'window size' input
    Running_Annual_Vol=np.sqrt(Running_Variance)*np.sqrt(252/frequency)  #Calculate Running Annual Vol

    #Place NEW Sampled data (prices) and Running Vols in DataFrame
    DF=pd.DataFrame(NEW_Sample)
    DF['Running_Vol']=Running_Annual_Vol
    
    #Create Plot
    DF.Price.plot()
    plt.legend()
    plt.ylabel('Yield (%)')
    DF.Running_Vol.plot(secondary_y=True, style='g',rot=90)
    plt.xlabel('Date')
    plt.ylabel('Running Vol') 
    plt.title('10 Year Bund Yield vs Annualized Running Vol (Window Size=200)')
    plt.legend(bbox_to_anchor=(0.8, 1))
    plt.text(0.8, 4.7, "Frequency={}. Window Size={}. Trading Hours per Day={}".format(frequency, window,trading_hours_per_day))

    
    return plt


# In[774]:


#Plots running annual vol vs price using daily data (extracted from hourly data)
#Considers sampling every 2 days, window size of 20 and 8 trading hours in a day
CHART_Running_Annual_Vol_with_Daily_Samples(2,50,8)


# In[ ]:





# #### e)iii) DAILY SAMPLING ON SPECIFIC TIME OF DAY

# In[767]:


def CHART_Running_Annual_Vol_with_Daily_Samples_on_Specific_Time_of_Day(frequency,sampling_time,window,trading_hours_per_day):
    Original_DAILY_Sample=data['Price'][sampling_time-1::trading_hours_per_day] #Grabs the Hourly Data and Converts it into Daily Data based on Sampling Time of Day AND Trading Hours per Day 
    NEW_Sample=Original_DAILY_Sample[frequency-1::frequency] #Creates New Sampling list based on sampling frequency input
    Returns=np.log(NEW_Sample) - np.log(NEW_Sample.shift(1)) #Calculates Returns on New Sample    
    Running_Variance=Returns.rolling(window).var() #Calculates daily running variance based on 'window size' input
    Running_Annual_Vol=np.sqrt(Running_Variance)*np.sqrt(252/frequency) 
    
    #Place NEW Sampled data (prices) and Running Vols in DataFrame
    DF=pd.DataFrame(NEW_Sample)
    DF['Running_Vol']=Running_Annual_Vol
    
    #Create Plot
    DF.Price.plot()
    plt.legend()
    #data.Price.plot()
    plt.ylabel('Yield (%)')
    DF.Running_Vol.plot(secondary_y=True, style='g',rot=90)
    plt.xlabel('Date')
    plt.ylabel('Running Vol') 
    plt.title('10 Year Bund Yield vs Annualized Running Vol ')
    plt.legend(bbox_to_anchor=(0.8, 1))
    plt.text(0.8, 3.5, "Sampling Time={}. Window Size={}. Trading Hours per Day={}".format(sampling_time, window,trading_hours_per_day))

    
    return plt


# In[768]:


#Plots running annual vol vs price using daily data (extracted from hourly data)
#Considers sampling every 2 days,samples made on the 5th hour of the day,
#window size of 50 and 8 trading hours in a day
CHART_Running_Annual_Vol_with_Daily_Samples_on_Specific_Time_of_Day(2,5,50,8)


# #### e)b) SAMPLING ONLY IF MOVE IS ≥ X bp

# In[761]:


def CHART_Running_Vol_with_Space_Increment_Sampling(increment_in_bp,window,trading_hours_per_day):
    Trading_Hours_in_Trading_Year=252*trading_hours_per_day #Calculates Trading Hours in a year
    Increment_Sample=[] #Create a List for the samples
    Dates_List=[] #Create list for Dates    
    
    Increment_Sample.append(data['Price'][0]) #Add the first data point in our data
    Dates_List.append(DATES[0:1]) #Add the first Date


    Index_List=[1] #Create a list for the index of the samples

    #Loop to add the element in the list ONLY if its value is more than X bp away from last sample
    counter=0 #counter to keep track in which hours did we sample
    for i in data['Price']:
        counter+=1
        if abs(i-np.array(Increment_Sample[-1:]))>=(increment_in_bp/100):
            Increment_Sample.append(i)
            Index_List.append(counter)
            Dates_List.append(DATES[counter])

        else:
            continue
    
    #Convert to a DataFrame
    Increment_Sample=pd.DataFrame(Increment_Sample)
    
    #Calculate Returns on NEW sample
    Returns=np.log(Increment_Sample) - np.log(Increment_Sample.shift(1))
    
    #Calculate Running average distance(in hours) between samples
    dt=pd.DataFrame([x - Index_List[i - 1] for i, x in enumerate(Index_List)][1:]).rolling(window).mean()
    
    # Calculate Running Variance
    Running_Variance=Returns.rolling(window).var() #Calculates hourly running variance based on 'window size' input
    
    #Calculate Annualized Vol based on average distance (in hours) between samples (dt) and the trading hours in a year
    Annual_Vol=np.sqrt(Running_Variance)*np.sqrt(Trading_Hours_in_Trading_Year/dt) 
    
    Final_DF=pd.DataFrame(Annual_Vol)#Create dataframe with Running Annual Vol
    Final_DF['Dates']=Dates_List   #Add sampled dates to the dataframe
    Final_DF.set_index('Dates',inplace=True, drop=True) #Set Dates as the index
    Final_DF.columns = ['Running_Annual_Vol']  #Assign column name

    #Create Plot
    Increment_Sample.columns = ['Price']  #Assign column name
    Increment_Sample.plot()
    plt.legend()
    plt.ylabel('Yield (%)')
    
    #plt.plot([], [], ' ', label="Extra label on the legend")
    #plt.legend()
    #data.Price.plot()
    
    Final_DF.Running_Annual_Vol.plot(secondary_y=True, style='g',rot=90)
    plt.xlabel('Date')
    plt.ylabel('Running Vol') 
    plt.title('10 Year Bund Yield vs Annualized Running Vol ')
    plt.legend(bbox_to_anchor=(.8, 1)) 
    plt.text(0.8, 7.2, "SAMPLING ONLY IF MOVE IS ≥ {} bp. Window Size= {}".format(increment_in_bp, window))
        
    return plt 


# In[762]:


CHART_Running_Vol_with_Space_Increment_Sampling(5,20,8)


# ## End of Project

# In[ ]:




