
 🚀 Predictive Maintenance with NASA CMAPSS Dataset


This project implements an end-to-end machine learning pipeline for predictive maintenance.

Given multivariate time-series sensor data from multiple machines, the objective is to:

- Detect degradation patterns  
- Estimate the Remaining Useful Life (RUL)  
- Predict whether a failure is imminent  



 ## 📊 Dataset

The project uses the NASA CMAPSS dataset:
- Multivariate time series
- 21 sensors
- Multiple machines (units)
- Run-to-failure data

Each engine degrades over time until failure.


## Approach
- Feature engineering (rolling mean, trends)  
- Target: failure within 30 cycles  
- Model: Random Forest  
- Metrics: Accuracy, Recall  


## 🖥️ Dashboard

Interactive Streamlit app for:

- Dataset selection (test only)  
- Machine (unit) selection  
- Sensor visualization  
- Failure prediction  
- Risk monitoring + alert  



## ⚙️ Usage

pip install -r requirements.txt

#### Training:

- Single dataset:

 python src/training.py --file train_FD001.txt

- All datasets: 

python src/training.py


#### Prediction

python src/predict.py --file test_FD001.txt


#### Dashboard:

streamlit run app/streamlit.py


## Notes

- Testing data is used for predictions (realistic scenario)
- Dataset in umbalanced


## License
MIT



