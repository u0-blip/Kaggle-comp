# Kaggle-comp

This is for various data science project that I did.


## Digit recognition
Used Keras, resulting in 404/2971 in leaderboard. 

## House price prediction
Used XGboost, resulting in 1521/5120 in leaderboard.


## Covid prediction
<image src='https://github.com/u0-blip/Kaggle-comp/blob/master/covid_predict/115929413_604322313605960_5763077778063753580_n.png?raw=true'>
Number of tests administrated. 
<image src='https://github.com/u0-blip/Kaggle-comp/blob/master/covid_predict/116425164_305081964237548_6531073777397213072_n.png?raw=true'>
The percentage of positives in tests.
From the graph, we can see that, even though the government has ramped up tests on the population, the number of new cases is still on the rise. 

For the prediction of Covid cases in Victoria. 
Firstly, I use naive SARIMA model from statsmodels. The model performed quite well. 
<image src='https://github.com/u0-blip/Kaggle-comp/blob/master/covid_predict/Figure_2.png?raw=true'>
But the accuracy is not as high as I was hoping to achieve.
So I used more information including the daily testing amount. 
<image src='https://github.com/u0-blip/Kaggle-comp/blob/master/covid_predict/Figure_1.png?raw=true'>
However, the resulting MSE increased from 65307.62 to 105680.79
I hypothesis this is due to the fact that the number of testing is in non-linear relationship with the new cases and thus cannot be modelled simplily as ARMA process. 
