""" 
I am understanding where white collar crime in San Francisco occurs and what are the
other variables that are correlated with it. To do this I will find more deeply analyze areas with high levels of
white collar crime and also perform a linear regression.
"""
__author__ = 'Jack King'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy import stats


# pandas imports the CSV as a Data Frame object
sfcrime = pd.read_csv("SFCrime__2018_to_Present.csv")
neighborhood_data = pd.read_csv("SF_Neighborhoods.csv", index_col="Neighborhood")
buisness_data = pd.read_csv("Map_of_Registered_Business_Locations.csv")

# only want white collar crime
wc = sfcrime[  (sfcrime['Incident Category'] == "Forgery And Counterfeiting") | (sfcrime['Incident Category'] == "Fraud") | (sfcrime['Incident Category'] == "Embezzlement")]

# Distribution of categories of white collar crime
wc["Incident Category"].value_counts().plot(kind='bar')
plt.ylabel('Number of Arrests')
plt.title("White Collar Crime in San Francisco")#
plt.xticks(rotation='horizontal')
plt.show()

# Now focusing on neighborhoods
wc_total = pd.crosstab(index=wc["Analysis Neighborhood"],
                            columns=wc["Incident Category"])

# Use boxplots to show how white collar crime is distributed across neighborhoods
def wcdistribution (type):
	wc_total.plot(kind = "box", y = type)
	plt.ylabel('Number of ' + type + ' Arrests')
	plt.title('Distribution of ' + type + ' Arrests by Neighborhood')
	plt.show()
wcdistribution("Forgery And Counterfeiting")
wcdistribution("Fraud")
wcdistribution("Embezzlement")

# bar chart to show more specifically where white collar crime occuring
wc_by_neigh = wc_total.sort_values("Fraud", ascending=False)
wc_by_neigh.plot(kind="bar")
plt.ylabel('Number of Arrests')
plt.title("White Collar Crime by Neighborhood")
plt.show()

# Add total column because from here on white collar crime will be grouped for easier Analysis
wc_total['Total'] = wc_total.apply(sum, axis=1)

# Find the z-scores of total white collar crime in each area
zscores1 = stats.zscore(wc_total)

# Combine new neighborhood data (population, Unemployment, median income, number of buisnesses) with crime data
# Clean data
wc_total = wc_total.drop(['Golden Gate Park', 'Hayes Valley', 'Japantown', 'Lone Mountain/USF', 'Lincoln Park', 'McLaren Park', 'Portola'])
buisness_data = buisness_data.dropna(subset=['Neighborhoods - Analysis Boundaries'])
buisness_data = buisness_data.rename(columns={"Neighborhoods - Analysis Boundaries": "Buisnesses"})
buis = buisness_data["Buisnesses"].value_counts()
buis = buis.drop(['Golden Gate Park', 'Hayes Valley', 'Japantown', 'Lone Mountain/USF', 'Lincoln Park', 'McLaren Park', 'Portola'])
# concatonate
wccrime = pd.concat( [neighborhood_data, wc_total, buis], axis = 1, sort=False)
# Need income as a numeric instead of a string
wccrime["Median Household Income ($)"] = pd.to_numeric(wccrime["Median Household Income ($)"])

# Show the distribtuions for the neighborhood data
def neighborhoodDist (neigh):
	wccrime.plot(kind = "box", y = neigh)
	plt.ylabel(neigh)
	plt.title('Distribution of ' + neigh + ' by Neighborhood')
	plt.show()
neighborhoodDist("Buisnesses")
neighborhoodDist("Population")
neighborhoodDist("Unemployment Rate (%)")
neighborhoodDist("Median Household Income ($)")

# get the z-scores of the neighborhood data for each Neighborhood
zscores2 = stats.zscore(wccrime)


# Use a linear regression to predict total white collar crime
y = wccrime["Total"]
x = wccrime[["Population", "Unemployment Rate (%)", "Median Household Income ($)", "Buisnesses"]]
lm = linear_model.LinearRegression()
model = lm.fit(x,y)
print(lm.score(x,y))
print(lm.coef_)
print(lm.intercept_)

# Graphing buinsesses versus total crime because the r-squared is high
plt.scatter(y=wccrime['Total'], x=wccrime['Buisnesses'])
plt.ylabel('White Collar Crime Arrests')
plt.xlabel('# of Buisnesses')
plt.title('White Collar Crime vs # of Buisnesses')
plt.show()

# Graphing income versus total crime because the r-squared is high
plt.scatter(y=wccrime['Total'], x=wccrime['Median Household Income ($)'])
plt.ylabel('White Collar Crime Arrests')
plt.xlabel('Median Household Income ($)')
plt.title('White Collar Crime vs Median Household Income')
plt.show()


# IS THERE AN INVERSE RELATIONSHIP BETWEEN REGULAR CRIME AND WHITE COLLAR

# repeat process except exclude instead of include white collar crime
notwc = sfcrime[  (sfcrime['Incident Category'] != "Forgery And Counterfeiting") & (sfcrime['Incident Category'] != "Fraud") & (sfcrime['Incident Category'] != "Embezzlement")]
notwc_total = pd.crosstab(index=notwc["Analysis Neighborhood"],
                            columns=notwc["Incident Category"])
notwc_total["Total_notwc"] = notwc_total.apply(sum, axis=1)
notwc_total = notwc_total.drop(['Golden Gate Park', 'Hayes Valley', 'Japantown', 'Lone Mountain/USF', 'Lincoln Park', 'McLaren Park', 'Portola'])

# concatonate
inverse = pd.concat( [notwc_total, wc_total], axis = 1, sort=False)

# scatter plot to show relationship
plt.scatter(y=inverse['Total'], x=inverse['Total_notwc'])
plt.ylabel('White Collar Crime Arrests')
plt.xlabel('Other Crime Arrests')
plt.title('White Collar Crime vs Other Crime')
plt.show()
