# Capstone Project- Fear and Sentiment in the Markets
**Cary Mosley, May 2020

## Project Overview

The goal of this project is to develop a weekly forecasting model for the S&P500 using the VIX index, a few exogenous measures of investor sentiment, and overall sentiment as calculated using New York Times headlines and article snippets. I will collect, clean, process and explore the data before building univariate, multivariate, and Long-Short Term Memory Neural Network forecasting models. I will first evaluate my models using RMSE and AIC/BIC before settling on the highest performing of each. Finally I will implement a couple different trading strategies over my test time period to see which model works best under potential real world trading implementations.

## Notebooks
Included in this github are a jupyter notebooks folder containing:

DataCollection.ipynb- Where I made API calls to NY Times and Quandl to collect sentiment data. I also scared Yahoo Finance for historical stock data.

DataCleanandProcess.ipynb- Where I cleaned and performed pre-processing on my sentiment data including NLP sentiment analysis.

EDA- A notebook where I performed significant EDA to explore my endogenous and exogenous variables.

TimeSeriesModeling.ipynb- I began with baseline models and continued through ARIMA, ARIMAX, VAR and VARMAX, and finally LSTM models.

TradingEvaluation.ipynb- I implement two different trading strategies and evaluate the outcome of my models here.

My presentation deck is located here: https://docs.google.com/presentation/d/1PaTaL7PmdYINP538glLY_h3aETjdScsV6ozQD3CedYA/edit?usp=sharing



## Glossary

I'm putting a few financial terms here for easy of explanation later

* Bearish: Thinking that the equity market is going to decrease in value

* Bullish: Thinking that the equity market is going to increase in value

* Long: Positive exposure to the equity market. Can be expressed in dollars or % leverage

* Short: Negative exposure to the equity market. Can be expressed in dollars or % leverage

* VIX Index- A Chicago Board of Options Exchange calculated index that measures the expected 30 day volatility in the S&P 500

## Executive Summary

I begin by collecting the daily closing prices for the SP500 and the VIX index. I also use the NY Times and the Quandl datasets containing the National Association of Active Managers sentiment and the American Associaton of Individual Investor sentiment data. From I cleaned and processed the data, including performing natural language processing sentiment analysis on the NY Times data before performing EDA. Next I will create progressively more complex models to examine the differences in how they perform. I go from a baseline persistance model to univariate models, to multivariate models, finally creating a LSTM model. The final evaluation I perform is seeing how the best performing model of each type does under two potential trading implementations. 

Under both implementations the LSTM model performed the best, resulting in signficant outperformance compared to both a buy and hold strategy as well as the other models. By performing well under the constant leverage and even better under the scaled leverage model, it is clear that the LSTM is performing quite well. The LSTM model performed even better under the significantly changing market condiitons that occurred during march and april 2020.

## Data Collection

I collected data from 6/30/2006 until 4/30/2020 due to the availability of my sentiment indicators. The nytimes article database goes back further but the quandl datasets stopped there. I scraped Yahoo Finance to grab the historical stock data. Next I used the NY Times API to collect all headline and snippet data going back through this time period. I didn't prefilter and ended up with over 1m articles. Finally I used the quandl API to collect my NAAIM and AAII sentiment data.

## Data Cleaning and Pre-processing

The first thing I did was to use a set of keywords to filter down the NY Times article dataframe. I ended up with approximately 35,000 entries over my time period. I then lemmatized, tokenized and removed stopwords from my article data. In the EDA notebook I'll use a couple of NLP toolkits to create a sentiment analysis for my data. I'll also take a weekly average of the sentiment data to keep it in the same format as the rest of my time series data.

Next, I cleaned the stock closing price data. I converted the daily closes to a weekly average over my time period to help smooth out the variation and to line it up with the weekly release of my sentiment indicators.

For the sentiment data the main processing I had to do was adjust the dates so that the weekly data would all sync up. 

## EDA

### Stock Prices
The first thing I did was take a look at the 4 different indices over my time period.


![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/combined_stock_time_series_final.png)


There has been a strong upward trend in the S&P500, the Nasdaq Index, and the Russell 2000 over the time period barring a few periods of downwrd moving. The VIX doesn't exhibit a long-run trend but there are a few period where it spikes. Due to the strong correlation between the S&P500, Nasdaq and Russell 2000 I will focus on just the S&P500 and the VIX index going forward.

### Sentiment Indexes

![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/AAII_time_series.png)

Although there is a lot of oscillation there doesnt appear to be any strong trends in the data. Since all three sum to 100% I won't use the neutral sentiment value going forward. As the neutral value can be computed from the Bullish and Bearish percentages I'm going to drop it. This oscillation is found in all of these indicators along with a lack of noticeable trend over time.


![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/leverage_time_series.png)

We can see that at times the most short manager still maintains a positive delta exposure to the market! The least long the most bullish manager gets appears to be below 100 only very infrequently. It also rarely gets above 200% exposure.


![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/sentiment_distributions.png)

Observations: 
* The % of investors that are Bullish looks pretty normal distributed centered around 30-40%.

* The % of investors that are beatish is skewed a bit lower but not too tailed. We can see that there are times where investors have been significantly more bearish than bullish.

* The spread between the two is roughly normal centered around positive 10% in the spread.

* The vast majority of the time the most bullish manager holds 200% leverage. It is rare that they are less than this. I might transform this into a dummy feature with 1 when the most bullish is 200% and 0 otherwise.

* The most bearish manager tends to be clustered at 0% long, -50%, -100%, -125%, -150% and -200%.

* The % leverage of the average manager is roughly normal with a  bit of a left tail and looks to be centered around 80%

* The distribution of the median manager is quite different with clusters at 50%, 80% and 100% and a clear left tail in times of market turmoil.

### NY Times Article EDA

#### Wordclouds
![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/article_count.png)

Most days have a single digit number of articles however some days tend to have a lot of articles. I imagine that the days with more articles are likely to be due to times of stress in the market so I plan to add article count as an engineered feature.

![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/headlines_cloud.png)

Above is the wordcloud from the article headlines and below is the wordcloud from the article snippets.

![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/snippets_cloud.png)

The word clouds of the headlines look somewhat similar with New York prominent in both, company very prominent in the snippet cloud. "Deal", "rise", "new", "sell", "investor" are some other common words. 

#### Sentiment Analysis

Next I used the NLTK TextBlob function as well as the VADER sentiment function to turn my headlines and snippets into numerical sentiment values. I then grouped and averaged by the same weekly periods as the rest of my data.

![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/nyt_sentiment_time_series.png)

The weekly average sentiment for both TextBlob and VADER as well as snippet and headline is quite noisy and doesnt seem to exhibit any strong trend.

![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/nyt_sentiment_distribution.png)

The sentiment data all looks roughly normal. The headlines for both TextBlob and VADER seem to be somewhat more negative than the snippets which are centered above zero.

## Initial Modeling

### Outline

* First I will check for stationarity/unit roots using the Dickey-Fuller test and if I find the time series are not stationary I will work with the data to arrive at stationarity.
* I will construct a baseline persistance model for both SPY and VIX
* I will construct ARIMA and then ARIMAX models for both SPY and VIX seperately using sentiment indicators and the NY Times sentiment analysis seperately
* I will work with my sentiment indicators to feature engineer a couple new features
* I will create a VAR and VARIMAX models using SPY and VIX together

### Stationarity

The first thing I did was check for unit roots/stationarity using the Dickey-Fuller test. For the SPY data I got a test stat of 0.22 and a p-value of .974. Thus I could not reject the null that the data is not stationary. I took a first order difference of the spy closing prices and performed the test again. The differenced test stat was -9.7 with a p-value of ~0. Thus we can reject the null that the data is not stationary. For VIX the closing prices resulted in a test stat of -3.04 with a p-value of 0.03. Thus I rejected the null that the data is not stationary. I also plan to use differenced VIX data as well as the closing prices later on.

![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/differenced_spy.png)

The differenced SPY values look stationary although its clear that there was a large increase in movement right at the end of my dataset.

![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/acf_pacf_spy.png)

The autocorrelation plot tells us whether we need to add MA terms while the PACF tells us about the AR terms. It is clear that the first lag is significant but somewhat weakly correlated in the ACF plot. The PACF plot shows us also that the 1st lag is likely to be somewhat useful to include as the AR term. For this first model I'll make sure to use a (1,1,1) and will try some other combinations as well. These plots look typical for a random walk time series in that there is some low correlation for the first lag but otherwise not much.

![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/acf_pacf_vix.png)

We can see from the ACF plot that there are a number of strongly positively correlated lags we can use for the moving average model. This is somewhat expected as there is a tendency for VIX to be sticky for a period of time although it does tend to mean revert to a white noise series. At times when VIX is in a period of turbulance the variance is likely to be highly correlated with itself. Later on I'll explore some ARCH and GARCH models to better model this behavior. The PACF shows us that only the first lag is strongly correlated. Thus a (1,0,n) model is likely to be decent with n potentially being a number of variables.

### Baseline Modles

For my baseline SPY and VIX models I will be using a persistance algorithm where the predicted value at the t+1 time step is the value at the t-1 time step. First I will set the train test date split I plan to use going forward, with the test data being the last year of my datset. I got Test RMSE values of 6.694 for the SPY model and 4.34 for the VIX model.

### ARIMA Models

First I built univariate models for both SPY and VIX before expanding them to include my exogenous sentiment features. The (1,1,1) ARIMA model for both SPY and VIX resulted in the lowest test RMSE levels although they were very close to my baseline models. This is disappointing but not unexpected due to the stock market being understood as a random walk model.

![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/spy_diff_arima.png)


![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/spy_forecast_arima.png)

Looking at this its clear that the model is essentially predicting very little week over week change. This leads to a upward trend for the forecasted SPY prices.

![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/spy_joint.png)

The vast majority of the scatterplot is packed relatively evenly around 0,0. This implies that knowing the change the day before is essentially uncorrelated with the next change. This is what we woudl expect from a random walk time series. Due to this low level of impact the model is unable to predict much from the previous times values.

![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/vix_diff_arima.png)


![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/vix_forecast_arima.png)

We get very similar results for the VIX forecasts. The model is vastly understating the realized variance and predicts close to a straight line trend. Now I'll add the exogenous variables.

### ARIMAX Models

The first thing I did was run through a (1,1,1) model using each of my sentiment features individually. The model that performed best was the VADER Sentiment model using the NY Times snippets. This model was marginally improved compared to the ARIMA and baseline but again was not particularly different. For VIX it looks like median active manager leverage was the best model by RMSE. Again this model was slightly improved compared to the ARIMA but it is not noticeably better than the baseline. Going forward, I'll use the median levarage and the VADER sentiment of the NY Times snippet as my exogenous variables.

![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/spy_diff_arimax.png)


![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/spy_forecast_arimax.png)

The ARIMAX predictions are a bit more volatile than the ARIMA but are still vastly understating the real variance.

![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/vix_diff_arimax.png)


![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/vix_forecast_arimax.png)

The forecast model looks like an improvement over the baseline and the ARIMA model. When there is a higher median leverage we would expect a lower value of VIX. This makes sense as VIX is known as the "fear" index.

### Feature Engineering

In this section I created two new features from my exogenous variables to see if I can improve my results. The features I plan to create are:

* Median Leverage * Bull-Bear Spread -If the leverage is higher and the spread is skewed Bullish this could be a positive signal
* Median Leverage * Snippet_Vader -If the leverage is higher and the sentiment is higher this could be a positive signal

### Multivariate Models
Now that I've created a couple new features I'll move on to multivariate time series analysis.

#### Granger Causality

First I performed a Granger's causality test and found that for 3 lags its right at the threshold p-value of .05. The other values are somewhat above the threshold so it is not clear that we can determine causality here.

#### Johansen's Cointegration

Using Johansen's cointegration test we reject the null that SPY and VIX differenced are not cointegrated.

#### VAR and VARMAX

First I performed a VAR model with a single exogenous varialbe. I examined multiple lags and found that a 1 lag model with my engineered feature of the median leverage times the bull-bear spread produced the best model. A model including all the exogenous variables performed slightly worse than the single feature model. Finally I tested a VARMAX model using and got a very mariginally improved result.

![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/var_diff.png)

Looking at the charts it appears that the multivariate models are performing better but still significantly underestimating the realized variance.

![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/var_forecast.png)

## Conclusions

Although my models managed to eek out an improvement over baseline they are not noticeable better and I would not recommend making any investing decisions based on them.