#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


Data={'S.No':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],'Percentage in Grade 10':[62.00,76.33,72.00,60.00,61.00,55.00,70.00,68.00,82.80,59.00,58.00,60.00,66.00,83.00,68.00,37.33,79.33,68.40,70.00,59.00,63.00,50.00,69.00,52.00,49.00],
      'Salary':[270000,200000,240000,250000,180000,300000,260000,235000,425000,240000,250000,180000,428000,450000,300000,240000,252000,280000,231000,224000,120000,260000,300000,120000,120000]}
mba_salary_df=pd.DataFrame(Data)
mba_salary_df


# In[4]:


mba_salary_df.head(10)


# In[5]:


mba_salary_df.info()


# In[6]:


import statsmodels.api as sm
X=sm.add_constant(mba_salary_df['Percentage in Grade 10'])
X.head(5)


# In[7]:


Y=mba_salary_df['Salary']


# In[8]:


from sklearn.model_selection import train_test_split  


# In[9]:


train_X,test_X,train_Y,test_Y=train_test_split(X,Y,train_size=0.8,random_state=100)
train_X,test_X,train_Y,test_Y


# In[10]:


mba_salary_lm = sm.OLS(train_Y,train_X).fit()


# In[11]:


print(mba_salary_lm .params)


# In[12]:


# for every 1% increase in grade 10 mba salary increases by 3244.473


# In[13]:


mba_salary_lm .summary()


# In[14]:


# Above model R-squared explain 18.1% of variation in salary.
# p value of test is 0.061 > 0.05 thus there is statistically insignificant relationship betwen feature, percentage in grade 10
#and mba salary.
# F statistic value(0.0612)is also greater than 0.05, thus overall model is also statistically insignififcant.


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


mba_salary_lm.resid


# In[17]:


mba_salary_resid=mba_salary_lm.resid


# In[18]:


probplot=sm.ProbPlot(mba_salary_resid)
plt.figure(figsize=(8,6))
probplot.ppplot(line='45')
plt.show()


# In[19]:


# It is important validate regression model to ensure its validity and goodness of the fit before it can be used for practical
#application.
# THE FOLLOWING CAN BE VALIDATE USING SIMPLE LINEAR REGRESSION:
#1.Co efficient of determination(R squared)
#2.Hypothesis test for regression coefficient .
#3.Analysis of variance for overall model validity(important for multiplelinaer regression)
#4.Residual analysis to validate regression model assumptions.
#5.Outlier analysis, its presence can significantly impact regression paarmeters. 


# In[20]:


#1. R SQUARED ( co efficient of determination) measures the percentage of variation in Y explained by the knowledge of X.
#2. R_squared value= SSR/SST = 1-(SSE/SST), where SST is the sum of squares of total variation, SSR is sum of squares of explained
#errors , SSE is sum of squares of unexplained variation
#3. Value of r squared lies btw 0 and 1.
#4.R squared is square of pearson correlation coefficient .
#3. Higher r squared indicates better fit.


# In[21]:


#1. ANOVA tests the relationship between a categorical and a numeric variable by testing the differences between two or more means
#This test produces a p-value to determine whether the relationship is significant or not.
#2.The F-statistic is given by: F=MSR/MSE , MSR is mean squared regression and MSE is mean squared error.
# F test is used for checking whwether overall regression model is statistically significant or not.
#3. In simple liner regression , p value for t-test anf F-test will be same since the null hypothesis is the same. 


# In[22]:


# RESIDUAL ANALYSIS:Residual or errors are difference btw actual value of outcome variable and predicted value.
# Residual (error) analysis is important to check whether the assumptions of regression models have been satisfied.
# It is performed to check the following:
#1.Rresidual are normally distributed.
#2. Variance of residuals is constant(Homoscedasticity).
#3. The functional form of regression is correctly specified.
#4. There are no outliers.


# In[23]:


# Test for Homoscedasticity: 
#1.An important assumption of regression model is that the residuals have constant variance.
#2.It can be observed by residual plot, where plot is btw standardised residual values and standardised predicted values.


# In[24]:


# Heteroscedasticity: An non constant variance of residuals , indicated by funnel shape in residual plot.


# In[25]:


def get_standardized_values(vals):
    return (vals-vals.mean())/vals.std()


# In[26]:


plt.scatter(get_standardized_values(mba_salary_lm.fittedvalues),get_standardized_values(mba_salary_resid))
plt.title("FIG: Residual Plot: MBA Salary Prediction");
plt.xlabel(" Standardized predicted values")
plt.ylabel(" Standardized residual ")


# In[27]:


# Insights of above plot-Residuals are random and have no funnel shape shows residuals have constant variance ( homosedacticity).


# In[28]:


# OUTLIER ANALYSIS: Outliers are observation whose values show a large deviation from mean value.
# It have significant influence on values of regression coefficient.
# Following distance measures are useful in identifying influenctial obsrvations:
#1. Z score
#2. Mahalonobis Distance
#3. Cooks distance 
#4. Leverage values


# In[29]:


# Z SCORE - It is standardized distance of an observation from its mean value.
# Z score of more than 3 may be flagged as on outlier.


# In[30]:


from scipy.stats import zscore


# In[31]:


mba_salary_df['z_score_salary'] = zscore(mba_salary_df.Salary)
mba_salary_df['z_score_salary']


# In[32]:


mba_salary_df[(mba_salary_df.z_score_salary>3.0)|(mba_salary_df.z_score_salary<-3.0)]


# In[33]:


# There is no observation that are outliers as per Z Score.


# In[34]:


# COOKS DISTANCE: 1.It is a measure how much predicted value of dependent variable change for all observation in sample when 
#a particular observation is excluded from sample of estimation.
#2. cooks distance value of more than 1 indicates highly influential observations


# In[35]:


import numpy as np 
mba_influence=mba_salary_lm.get_influence()
(c,p)=mba_influence.cooks_distance


# In[36]:


(c,p)


# In[37]:


plt.stem(np.arange(len(train_X)),np.round(c,2))
plt.title("Figure : cooks distance for all observation in MBA salary dataset")
plt.xlabel(" Row index")
plt.ylabel(" Cooks Distance")


# In[39]:


# No observation abpve 1 so none of them are outliers in above plot.


# In[40]:


# LEVERAGE VALUES: It measures influence of that observation on overall fit of regression function and is related to Mahalanobis
#distance.


# In[44]:


from statsmodels.graphics.regressionplots import influence_plot
fig, ax= plt.subplots(figsize=(8,6))
influence_plot(mba_salary_lm, ax=ax)
plt.title("Figure: Leverage values Vs Residuals")
plt.show()


# In[ ]:


# Insight of above plot-
# The size off circle is proportional to residual and leverage value.Larger circle, larger the residual hence large inluence.

