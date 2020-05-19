# Capstone Project- Fear and Sentiment in the Markets
**Cary Mosley, May 2020

## Notebooks
Included in this github are a jupyter notebooks folder containing:

DataCollection.ipynb- Where I made API calls to NY Times and Quandl to collect sentiment data. I also scared Yahoo Finance for historical stock data.

DataCleanandProcess.ipynb- Where I cleaned and performed pre-processing on my sentiment data including NLP sentiment analysis.

EDA- A notebook where I performed significant EDA to explore my endogenous and exogenous variables.

TimeSeriesModeling.ipynb- A notebook where I began with baseline models and continued through ARIMA, ARIMAX, VAR and VARMAX models on the SP500 and VIX indices.

My presentation deck is located here: https://docs.google.com/presentation/d/1PaTaL7PmdYINP538glLY_h3aETjdScsV6ozQD3CedYA/edit?usp=sharing

## Goals

The goal of this project is to combine SP500 closing data and VIX index closing data with various sentiment indicators to see if I can create a model that performs better than a baseline model. The VIX index is generally considered a fear index so the hope is that the mood (sentiment), as measured by various sentiment indicators, of investors might have an ability to improve the forecasting of the SP500.

## Executive Summary

I begin by collecting the daily closing prices for the SP500 and the VIX index. I also use the NY Times and the Quandl datasets containing the National Association of Active Managers sentiment and the American Associaton of Individual Investor sentiment data. From here I will clean and process the data, including performing natural language processing sentiment analysis on the NY Times data before performing EDA. Once this is complete I will go step by step beginning with a baseline model and then progressing to an ARIMA model. Next I will add my exogenous variables and move on to an ARIMAX and then a VARIMAX model. 

I ended up with a VARMAX model that uses differenced SPY and VIX prices, with 1 lag and using the Median active manager leverage as my best model and models that very slightly outperformed my baseline. This is not too surprising as there is an entire industry focused on predicted financial market returns that has not been able to solve the problem. 

## Data Collection

I collected data from 6/30/2006 until 4/30/2020 due to the availability of my sentiment indicators. The nytimes article database goes back further but the quandl datasets stopped there. I scraped Yahoo Finance to grab the historical stock data. Next I used the NY Times API to collect all headline and snippet data going back through this time period. I didn't prefilter and ended up with over 1m articles. Finally I used the quandl API to collect my NAAIM and AAII sentiment data.

## Data Cleaning and Pre-processing

The first thing I did was to use a set of keywords to filter down the NY Times article dataframe. I ended up with approximately 35,000 entries over my time period. I then lemmatized, tokenized and removed stopwords from my article data. In the EDA notebook I'll use a couple of NLP toolkits to create a sentiment analysis for my data. I'll also take a weekly average of the sentiment data to keep it in the same format as the rest of my time series data.

Next, I cleaned the stock closing price data. I converted the daily closes to a weekly average over my time period to help smooth out the variation and to line it up with the weekly release of my sentiment indicators.

For the sentiment data the main processing I had to do was adjust the dates so that the weekly data would all sync up. 

## EDA

### Stock Prices
The first thing I did was take a look at the 4 different indices over my time period.

![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/stock_time_series_all.png)

![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/combined_stock_time_series.png)


There has been a strong upward trend in the S&P500, the Nasdaq Index, and the Russell 2000 over the time period barring a few periods of downwrd moving. The VIX doesn't exhibit a long-run trend but there are a few period where it spikes. Due to the strong correlation between the S&P500, Nasdaq and Russell 2000 I will focus on just the S&P500 and the VIX index going forward.

### Sentiment Indexes

![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/AAII_time_series.png)

Although there is a lot of oscillation there doesnt appear to be any strong trends in the data. Since all three sum to 100% I won't use the neutral sentiment value going forward. As the neutral value can be computed from the Bullish and Bearish percentages I'm going to drop it.

![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/spread_time_series.png)

I'm planning to keep the spread between the % investors that are bullish and bearish in my data set. As expected there is a large amount of oscillation but no strong trend in the time series.

![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/leverage_time_series.png)

We can see that at times the most short manager still maintains a positive delta exposure to the market! The least long the most bullish manager gets appears to be below 100 only very infrequently. It also rarely gets above 200% exposure.

![models](https://github.com/CaryMosley/FinalProject/blob/CaryM/Images/mean_median_leverage_time_series.png)

The mean and median leverage percentage also tends to oscillate without any clear trend with the average exposure genereally lower than the median.

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
