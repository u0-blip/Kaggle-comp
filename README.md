# Kaggle-comp

This is for various data science project that I did.




## Covid prediction
For the prediction of Covid cases in Victoria. 
Firstly, I use naive SARIMA model from statsmodels. The model performed quite well. 
<image src='https://github.com/u0-blip/Kaggle-comp/blob/master/covid_predict/Figure_2.png?raw=true'>
But the accuracy is not as high as I was hoping to achieve.
So I used more information including the daily testing amount. 
<image src='https://github.com/u0-blip/Kaggle-comp/blob/master/covid_predict/Figure_1.png?raw=true'>
However, the resulting MSE increased from 65307.62 to 105680.79
I hypothesis this is due to the fact that the number of testing is in non-linear relationship with the new cases and thus cannot be modelled simplily as ARMA process. 
