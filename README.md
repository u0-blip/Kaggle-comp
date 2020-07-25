# Kaggle-comp

This is for various data science project that I did.




## Covid prediction
For the prediction of Covid cases in Victoria. 
Firstly, I use naive SARIMA model from statsmodels. The model performed quite well. 
<image src='https://github.com/u0-blip/Kaggle-comp/blob/master/covid_predict/Figure_2.png?raw=true'>
But the accuracy is not as high as I was hoping to achieve.
So I used more information including the daily testing amount. 
<image src='https://github.com/u0-blip/Kaggle-comp/blob/master/covid_predict/Figure_1.png?raw=true'>
As can be seen in the graph, the accuracy of the prediction is dramatically increased.
This demonstrated that testing is an important factor in the number of case detected. 

