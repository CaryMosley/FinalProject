# Capstone Project- Fear and Sentiment in the Markets
Cary Mosley, May 2020

## Project Overview

The goal of this project is to develop a weekly forecasting model for the S&P500 using the VIX index, a few exogenous measures of investor sentiment, and overall sentiment as calculated using New York Times headlines and article snippets. I will collect, clean, process and explore the data before building univariate, multivariate, and Long-Short Term Memory Neural Network forecasting models. I will first evaluate my models using RMSE and AIC/BIC before settling on the highest performing of each. Finally I will implement a couple different trading strategies over my test time period to see which model works best under potential real world trading implementations.

My presentation deck is located here: 
https://docs.google.com/presentation/d/14GSWN8WfCmeFLZ31IysbL6ILkZh_dmk1Hl8Qw0H3660/edit?usp=sharing

## Notebooks
Included in this github are a jupyter notebooks folder containing:

DataCollection.ipynb- Where I made API calls to NY Times and Quandl to collect sentiment data. I also scared Yahoo Finance for historical stock data.

DataCleanandProcess.ipynb- Where I cleaned and performed pre-processing on my sentiment data including NLP sentiment analysis.

EDA- A notebook where I performed significant EDA to explore my endogenous and exogenous variables.

TimeSeriesModeling.ipynb- I began with baseline models and continued through ARIMA, ARIMAX, VAR and VARMAX, and finally LSTM models.

TradingEvaluation.ipynb- I implement two different trading strategies and evaluate the outcome of my models here.




## Glossary

I'm putting a few financial terms here for easy of explanation later

* Bearish: Thinking that the equity market is going to decrease in value

* Bullish: Thinking that the equity market is going to increase in value

* Long: Positive exposure to the equity market. Can be expressed in dollars or % leverage

* Short: Negative exposure to the equity market. Can be expressed in dollars or % leverage

* VIX Index- A Chicago Board of Options Exchange calculated index that measures the expected 30 day volatility in the S&P 500

## Executive Summary

I begin by collecting the daily closing prices for the SP500 and the VIX index. I also use the NY Times and the Quandl datasets containing the National Association of Active Managers sentiment and the American Associaton of Individual Investor sentiment data. From I cleaned and processed the data, including performing natural language processing sentiment analysis on the NY Times data before performing EDA. Next I will create progressively more complex models to examine the differences in how they perform. I go from a baseline persistance model to univariate models, to multivariate models, finally creating a LSTM model. For each of my models I use RMSE/AIC/BIC and cross validation to choose the best lag terms and feature set. The final evaluation I perform is seeing how the best performing model of each type does under two potential trading implementations. 

Under both implementations the LSTM model performed the best, resulting in signficant outperformance compared to both a buy and hold strategy as well as the other models. By performing well under the constant leverage and even better under the scaled leverage model, it is clear that the LSTM has robust performance. The LSTM model did even better under the significantly changing market condiitons that occurred during march and april 2020.

## Data Collection

I collected data from 6/30/2006 until 4/30/2020 due to the availability of my sentiment indicators. The nytimes article database goes back further but the quandl datasets stopped there. I scraped Yahoo Finance to grab the historical stock data. Next I used the NY Times API to collect all headline and snippet data going back through this time period. I didn't prefilter and ended up with over 1m articles. Finally I used the quandl API to collect my NAAIM and AAII sentiment data.

## Data Cleaning and Pre-processing

The first thing I did was to use a set of keywords to filter down the NY Times article dataframe. I ended up with approximately 35,000 entries over my time period. I then lemmatized, tokenized and removed stopwords from my article data. In the EDA notebook I'll use a couple of NLP toolkits to create a sentiment analysis for my data. I'll also take a weekly average of the sentiment data to keep it in the same format as the rest of my time series data.

Next, I cleaned the stock closing price data. I converted the daily closes to a weekly average over my time period to help smooth out the variation and to line it up with the weekly release of my sentiment indicators.

For the sentiment data the main processing I had to do was adjust the dates so that the weekly data would all sync up. 

## EDA

### Stock Prices
The first thing I did was take a look at the 4 different indices over my time period.


![models](https://github.com/CaryMosley/FinalProject/blob/master/Images/combined_stock_time_series_final.png)


There has been a strong upward trend in the S&P500, the Nasdaq Index, and the Russell 2000 over the time period barring a few periods of downwrd moving. The VIX doesn't exhibit a long-run trend but there are a few period where it spikes. Due to the strong correlation between the S&P500, Nasdaq and Russell 2000 I will focus on just the S&P500 and the VIX index going forward.

### Sentiment Indexes

![models](https://github.com/CaryMosley/FinalProject/blob/master/Images/AAII_time_series_final.png)

Although there is a lot of oscillation there doesnt appear to be any strong trends in the data. This oscillation is found in all of these indicators along with a lack of noticeable trend over time.


![models](https://github.com/CaryMosley/FinalProject/blob/master/Images/leverage_time_series_final.png)

We can see that at times the most short manager still maintains a positive delta exposure to the market! The least long the most bullish manager gets appears to be below 100 only very infrequently. It also rarely gets above 200% exposure.


![models](https://github.com/CaryMosley/FinalProject/blob/master/Images/sentiment_distributions_final.png)

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
![models](https://github.com/CaryMosley/FinalProject/blob/master/Images/article_count_final.png)

Most days have a single digit number of articles however some days tend to have a lot of articles. I imagine that the days with more articles are likely to be due to times of stress in the market so I plan to add article count as an engineered feature.

![models](https://github.com/CaryMosley/FinalProject/blob/master/Images/headlines_cloud_final.png)

Above is the wordcloud from the article headlines and below is the wordcloud from the article snippets.

![models](https://github.com/CaryMosley/FinalProject/blob/master/Images/snippets_cloud_final.png)

The word clouds of the headlines look somewhat similar with New York prominent in both, company very prominent in the snippet cloud. "Deal", "rise", "new", "sell", "investor" are some other common words. 

#### Sentiment Analysis

Next I used the NLTK TextBlob function as well as the VADER sentiment function to turn my headlines and snippets into numerical sentiment values. I then grouped and averaged by the same weekly periods as the rest of my data. The weekly average sentiment for both TextBlob and VADER as well as snippet and headline is quite noisy and doesnt seem to exhibit any strong trend.

![models](https://github.com/CaryMosley/FinalProject/blob/master/Images/nyt_sentiment_distribution_final.png)

The sentiment data all looks roughly normal. The headlines for both TextBlob and VADER seem to be somewhat more negative than the snippets which are centered above zero.

## Modeling

### Outline

* First I will check for stationarity/unit roots using the Dickey-Fuller test and if I find the time series are not stationary I will work with the data to arrive at stationarity.
* I then construct a baseline persistance model for both SPY and VIX
* I construct ARIMA and then ARIMAX models for both SPY and VIX seperately using sentiment indicators and the NY Times sentiment analysis seperately
* I featured engineered two new sentiment features
* I built multivariate VAR and VARIMAX models
* I created a Long Short-Term Memory Neural Net Model to predict SPY

### Stationarity

The first thing I did was check for unit roots/stationarity using the Dickey-Fuller test. For the SPY data I got a test stat of 0.22 and a p-value of .974. Thus I could not reject the null that the data is not stationary. I took a first order difference of the spy closing prices and performed the test again. The differenced test stat was -9.7 with a p-value of ~0. Thus we can reject the null that the data is not stationary. For VIX the closing prices resulted in a test stat of -3.04 with a p-value of 0.03. Thus I rejected the null that the data is not stationary. I also plan to use differenced VIX data as well as the closing prices later on.


### Baseline Modles

For my baseline SPY and VIX models I will be using a persistance algorithm where the predicted value at the t+1 time step is the value at the t-1 time step. First I will set the train test date splits I plan to use going forward. 

### ARIMA Models

First I built univariate models for both SPY and VIX before expanding them to include my exogenous sentiment features. The (1,1,1) ARIMA model for both SPY and VIX resulted in the lowest test RMSE levels although they were very close to my baseline models. This is disappointing but not unexpected due to the stock market being understood as a random walk model.

Looking at this its clear that the model is essentially predicting very little week over week change. This leads to a upward trend for the forecasted SPY prices.

We get very similar results for the VIX forecasts. The model is vastly understating the realized variance and predicts close to a straight line trend. Now I'll add the exogenous variables.

### ARIMAX Models

The first thing I did was run through a (1,1,1) model using each of my sentiment features individually. The model that performed best was the VADER Sentiment model using the NY Times snippets. This model was marginally improved compared to the ARIMA and baseline but again was not particularly different. For VIX it looks like median active manager leverage was the best model by RMSE. Again this model was slightly improved compared to the ARIMA but it is not noticeably better than the baseline. Going forward, I'll use the median levarage and the VADER sentiment of the NY Times snippet as my exogenous variables.

The ARIMAX predictions are a bit more volatile than the ARIMA but are still vastly understating the real variance. When there is a higher median leverage we would expect a lower value of VIX. This makes sense as VIX is known as the "fear" index.

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

First I performed a VAR model with a single exogenous variable. I examined multiple lags and a variety of feature sets. Finally I tested a VARMAX model using and got a very mariginally improved result. The multivariate models are performing better but still significantly underestimating the realized variance. The final models I'm moving towards the trading evaluation are a 1-lag VAR model and a (1,2) VARMAX model, both with my final exogenous feature set.

### LSTM Model

The final model I create was a LSTM Neural net model. In order to create this model I first scaled and then converted my endogenous and exogenous variables to supervised ones. I then used TensorFlow Keras to design and build my LSTM forecast model for the SP500.

Now that my models have been built I will implement two different trading strategies in my final notebook.

## Trading Evaluation

I decided to implement my forecasts via two different trading strategies and compare them to a simple buy and hold passive invesment strategy. For both my strategies, I look at the predicted change over the next time period and then calculate a return based on the actual change in SPY. For my first strategy, if the forecast is positive I take a 100% long position and if the forecast is negative I take a 100% short position. For my second strategy I scale my leverage based on the size of the expected forecast using the following rules:
* Predicted change <0.5% - 50% leverage
* Predicted change 0.5%<x<1% 100% leverage
* Predicted change >1% 200% leverage

### Results

My LSTM model performed the best under both trading implementations, significantly beating the buy and hold strategy.

![models](https://github.com/CaryMosley/FinalProject/blob/master/Images/trading_results_const_final.png)

The LSTM model performed the best with a significant outperformance compared to the other models under the first trading strategy. The ARIMAX model still performed quite well, beating the buy and hold strategy but the VAR/VARMAX models underperformed consistently. Both the LSTM and ARIMAX models seemed to do well when the market experienced downward shocks which could be an especially valuable result. Using the LSTM model wouldve resulted in a PnL of $50,000 vs $13,000 using buy and hold.

![models](https://github.com/CaryMosley/FinalProject/blob/master/Images/trading_results_scaled_final.png)

Again, the LSTM model outperformed the rest of the herd. By scaling the leverage based on the size of the expected move, I increased overall profit quite substantially. Note that the ARIMAX model performed worse under this leveraged implementation while all of the multivariate forecasts performed better. Although the ARIMAX model performed well directionally as evidenced by the 100% leverage model, it did not perform as well when the results were related to the strength of its predictions as measured by size. The scaled LSTM model seemed to do well when the market experienced downward shocks which could be an especially valuable result. Using the LSTM model wouldve resulted in a PnL of $90,000 vs $13,000 using buy and hold.


## Conclusions

My LSTM model beat the buy and hold trading strategy in both trading scenarios I developed. The model outperformed at almost every point in the testing time period as well as during a period of market turbulence. I plan to continue to monitor and update my LSTM model as time goes on as well as build individual security predictions to see how they work across a basket of names.