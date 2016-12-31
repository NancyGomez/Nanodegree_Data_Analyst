
# coding: utf-8

# # Titanic Data Analysis
# By Nancy Gomez

# ### Questions:
# General: Which factors made people more likely to survive?
# 1. Did gender affect survivablity? Which gender was most likely to survive?
# 2. Did age affect survivability? Which age group was most likely to survive?
# 3. Did class affect survivability? Which class was most likely to survive?

# ### Hypotheses:
# Alternative Hypothesis: If gender was related in any way to the survivability, then females will be more likely to survive compared to men.
# 
# Null Hypothesis: A person's gender didn't affect their survivability.

# Alternative Hypothesis: If age was related in any way to the survivability, then young adults (ages 20 - 30) were more likely to survive compared to all the other age groups.
# 
# Null Hypothesis: A person's age didn't affect their survivability.

# Alternative Hypothesis: If class was related in any way to survivability, then the wealthiest class would be more likely to survive.
# 
# Null Hypothesis: A person's class didn't affect their survivability.

# ### Data Wrangling Code:

# In[348]:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from pandas import DataFrame
get_ipython().magic(u'matplotlib inline')
data_df = pd.read_csv('titanic_data.csv')
data_df.head()


# ##### Cleaning up Data:
# After I checked the dataframe using the head() function, I saw the categories that the data offers and realized some were not necessary to keep. The data I figured would in no way affect survivability were Ticket, Name, Embarkment, Cabin, and Fare. I also noticed that some sections had the value "NaN," meaning that some data was missing. I decided I needed to check how many values were missing for the categories to establish my approach.

# ##### Helper Print Functions:
# All the helper print functions are located here for organizational purposes :)

# In order to check the empty data, I realized I could create a general function that takes in the data frame and checks for null values. It prints the total missing values as well as its corresponding percentage.

# In[349]:

def printEmptyData(dataframe, label):
    empty = pd.isnull(dataframe[label])
    empty_percent = len(empty[empty == True]) / float(len(dataframe[label])) * 100
    print len(empty[empty == True]), '= {0:.2f}%'.format(empty_percent)


# This function is used to print the total number of a categories start through end within the data in label from the dataframe 

# In[350]:

def printPortion(start, end, dataframe, label):
    for i in range (start, end):
        portion =  len(dataframe[label][dataframe[label] == i])
        percent = portion / float(len(dataframe[label])) * 100
        print '{} ({}): {} people, {:.2f}%'.format(label, i, portion, percent)


# In order to print the total number of survivors / deaths within a category (label) from the dataframe

# In[351]:

def printStats(start, end, dataframe, label):
    for i in range(start, end):
        survivors = dataframe[dataframe['Survived'] == 1]
        casualties = dataframe[dataframe['Survived'] == 0]
    
        label_survivors = len(survivors[survivors[label] == i])
        label_casualties = len(casualties[casualties[label] == i])
    
        s_percent = label_survivors / float(len(dataframe[dataframe[label] == i])) * 100
        c_percent = label_casualties / float(len(dataframe[dataframe[label] == i])) * 100

        print ("{} ({}) Survivors: {} people, {:.2f}%\tCasualties: {} people, {:.2f}%"
               .format(label, i, label_survivors, s_percent, label_casualties, c_percent)).expandtabs(25)


# In[352]:

printEmptyData(data_df, 'Sex')


# In[353]:

printEmptyData(data_df, 'Age')


# In[354]:

printEmptyData(data_df, 'Pclass')


# ##### Decision on Data:
# In terms of the missing data for the age, I decided that there is too much information missing within the age category to come to any solid conclusions. The percentage of missing data is nearly 20%, and since this percentage is so high, I disregarded using the mean substitution method (where all the null values are replaced with the overall mean of the data) since it would alter the data too much. Instead, to deal with the missing data, I plan to only observe the data that is present. Luckily, matplotlib takes care of this for me internally for numeric data.

# Regardless of the amount of data that is presented, I don't plan to use the Name, Ticket, Cabin, Embarkment, or Fare categories so I will delete these. I only plan to dwell into the Sex, Age, and Pclass, so I also decided to delete the SibSp and Parch categories. 

# In[355]:

del data_df['Cabin']; del data_df['Ticket']; del data_df['Fare'];
del data_df['SibSp']; del data_df['Parch']; del data_df['Embarked']


# #### Statistical Functions:
# Standardizing data is important to accurately interpret how far it is from the mean. The argument "ddof = 0" simply allows us to check the uncorrected standardized data (which is essential for calculating Pearson' R)

# In[356]:

def standardizeData(values):
    standardized_values = (values - values.mean()) / values.std(ddof = 0)
    return standardized_values


# In order to determine whether two data sets have a correlation or not, we can calculate Pearson's R and then calculate the 
# percentage of the correlation by taking the absolute value and moving the decimal point. Above 50% means the correlation is significant, and below is irrelevant.

# In[357]:

def correlation(values1, values2):
    std_1 = standardizeData(values1)
    std_2 = standardizeData(values2)
    pearsons_r = abs((std_1 * std_2).mean()) * 100
    return float('{0:.2f}'.format(pearsons_r))


# #### Visual Functions

# In[358]:

def graphBy(dataframe, label1, label2):
    for survive, group in dataframe.groupby(label1)[label2]:
        plt.figure(); group.hist()
        plt.xlabel(label2); plt.ylabel('Number of People')
        if (survive):
            plt.title('Survivors')
        else:
            plt.title('Casualties')


# In[359]:

def plotBySurvival(dataframe, label):
    for survive in range (0, 2):
        count = Counter(data_df[data_df['Survived'] == survive][label])
        df = pd.DataFrame.from_dict(count, orient='index')
        df.plot(kind='bar'); plt.xlabel(label); plt.ylabel('Number of People')
        if (survive):
            plt.title('Survivors')
        else:
            plt.title('Casualties')


# In[360]:

def graphPieBySurvival(dataframe, label, value):
    survivors = dataframe[dataframe['Survived'] == 1]
    casualties = dataframe[dataframe['Survived'] == 0]
    
    label_survivors = len(survivors[survivors[label] == value])
    label_casualties = len(casualties[casualties[label] == value])
    
    data = pd.Series([label_survivors, label_casualties],
                     index = [value.title() + ' Survivors',value.title() + ' Deaths'],
                     name = value.title())
    data.plot('pie')


# #### Age Survival Graphs:

# In[361]:

graphBy(data_df, 'Survived', 'Age')


# #### Sex Survival Graphs:

# In[362]:

graphPieBySurvival(data_df, 'Sex', 'male')


# In[363]:

graphPieBySurvival(data_df, 'Sex', 'female')


# #### Class Survival Graphs:

# In[364]:

plotBySurvival(data_df, 'Pclass')


# # Analysis:
# In order to check the correlation between the data, it must all be numerical. So I changed the strings to corresponding numeric values starting from 0.

# In[365]:

data_df['Sex'] = data_df['Sex'].replace({'female': 0}, regex = True)
data_df['Sex'] = data_df['Sex'].replace({'male': 1}, regex = True)


# ### Gender Results:

# In[366]:

correlation(data_df['Survived'], data_df['Sex'])


# Precentage of civilians from each gender:

# In[367]:

printPortion(0, 2, data_df, 'Sex')


# Percentage of Survivors and Casualties from each gender:

# In[368]:

printStats(0, 2, data_df, 'Sex')


# ###### Conclusion Based off Gender: 
# Since Pearson's R was about 54.3%, It's obvious that there definitely is a correlation between sex and survivability. It's not as high as I would have expected though. I originally believed that this value would be closer to 60 or 70 percent.
# 
# Due to this, I reject the null hypothesis, but must say that this does not by any means signify that a person's gender was a causation of their survivability. Instead, there was simply a noticeable correlation between the two. As noted with the printed statistics of the two genders (0 being female and 1 being male), we can easily see that 74% of the females managed to survive whereas only about 20% of the males managed to survive. Not to mention the fact that the males were a majority of the civilians on the Titanic, making the survival rate of women even more astounding.
# 
# I believe the reason the survival rate of women was so high due to the common saying "Women and children first." It was noble for a man to always prioritize the safety or comfort of women and children. It seems this has applied to the disasterous sinking of the Titanic.

# ### Age Results:

# In[369]:

correlation(data_df['Survived'], data_df['Age'])


# ###### Conclusion Based off Age: 
# Since Perason's R was only about 7.8%, it seems that the correlation between the two is practically non-existent. I also thought that this value would be much higher considering the graphs had shown some bulking within the 20-30 year range, I suppose the relationship was so small since both graphs had similar behaviors.
# 
# Due to this, I failed to reject the null hypothesis. The correlation between Age and Survivability was too faint. It should also be noted that this analysis is partially incomplete with the 177 missing data values.

# ### Class Results:

# In[370]:

correlation(data_df['Survived'], data_df['Pclass'])


# Precentage of civilians from each class:

# In[371]:

printPortion(1, 4, data_df, 'Pclass')


# Precentage of Survivors within their respective class:

# In[372]:

printStats(1, 4, data_df, 'Pclass')


# ##### Conclusion Based off Class:
# The correlation between the two data sets only turned out to be about 33.85% which isn't high enough for my standards to reject the Null Hypothesis, however the data within the statistics seems to pronounce the relationship between class and survival. The civilians within the first class had a 62% survival rate which is much higher than second class and third class. What is most shocking is that the people within the third class (although the majority) had the highest casualty rate at over 75%.
# 
# Reasons for this may include the fact that wealthy people were prioritized for life boats. The wealthy usually manage to get their way by bribing or simply by being important enough. Another reason may be that the location where each respective class resided in directly affected their ability to reach life boats. First class was located in the middle and upper levels of the ship, while third class was at the lower and edges of the boat. 

# ## Overall Conclusion
# 
# There were many limitations in the project which include the missing data, primarily from the age column but also from the Embarked column. This missing data causes incompleteness in the overall result but since this data was ommitted, it is simply a smaller data set than was orignally expected.
# 
# Also we musn't forget that there usually will always exist some sort of bias in the collection of data. Specifically for this data set, I believe that there could be errors within the collection of it. Since I am unaware of which method was used to retrieve the data, I can not be entirely sure of all the biases that may exist. I also don't believe that everyone who was on the Titanic is accounted for, perhaps people without families also boarded and were never deemed missing. There's also the potential of people who could have snuck on-board who are also not accounted for.
# 
# To guarantee accurate data is practically impossible, since there are so many variables which may interfere. Overall, within my sampling I feel very confident within my analysis. I also just wanted to add how fun this project was, I thoroughly enjoyed the thinking that was required of me in order to achieve all of these results.

# 
