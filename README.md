# IoT Anomaly Detection and Fault Forecasting with LSTM Autoencoder

## Overview
This repository implements a **hybrid deep learning approach** for fault detection and prediction using IoT sensor data. It combines **unsupervised anomaly detection** with an **LSTM-based forecasting model** to improve fault prediction accuracy.

## Key Techniques Used
- **LSTM Autoencoder for Anomaly Detection**
  - Encodes sensor data into a lower-dimensional representation and reconstructs it to detect deviations.
  - Uses reconstruction error as an anomaly score.

- **LSTM-based Fault Forecasting**
  - Incorporates extracted features and anomaly scores to predict potential faults.
  - Uses a **sigmoid-activated Dense layer** for binary fault classification.

- **Feature Engineering**
  - Computes statistical features like mean sensor values.
  - Appends anomaly scores to improve fault prediction performance.

## Libraries & Dependencies
- **TensorFlow/Keras** – Deep learning framework for LSTM models.
- **NumPy & Pandas** – Data manipulation and feature extraction.
- **Matplotlib & Seaborn** – Visualization of training loss, anomaly thresholds, and ROC curves.

## Dataset
The dataset used in this project is the [Intel Lab Data](https://db.csail.mit.edu/labdata/labdata.html), collected from 54 sensors deployed in the Intel Berkeley Research lab between February 28th and April 5th, 2004. The sensors recorded the following parameters:

- **Temperature**: Measured in degrees Celsius.
- **Humidity**: Temperature-corrected relative humidity (%).
- **Light**: Measured in Lux.
- **Voltage**: Battery voltage in volts.

Data was sampled every 31 seconds, resulting in approximately 2.3 million readings. Each record includes:

- `date`: Recording date (`yyyy-mm-dd`).
- `time`: Recording time (`hh:mm:ss.xxx`).
- `epoch`: Monotonically increasing sequence number from each sensor.
- `moteid`: Sensor ID (ranging from 1 to 54).
- `temperature`: Recorded temperature.
- `humidity`: Recorded humidity.
- `light`: Recorded light intensity.
- `voltage`: Recorded battery voltage.

For more details and to access the dataset, visit the [Intel Lab Data page](https://db.csail.mit.edu/labdata/labdata.html).

## Performance Metrics
- **Anomaly Detection**
  - Threshold-based anomaly flagging.
  - The following table outlines the relationship between reconstruction error percentiles and the corresponding percentage of anomalies detected:

| Percentile | Reconstruction Error Threshold | Percentage of Anomalies Detected |
|------------|-------------------------------:|---------------------------------:|
| 90th       | 0.010597                       | 10.00%                           |
| 92nd       | 0.013420                       | 8.02%                            |
| 94th       | 0.016075                       | 6.01%                            |
| 95th       | 0.016975                       | 5.00%                            |
| 96th       | 0.018287                       | 4.01%                            |
| 97th       | 0.019191                       | 3.02%                            |
| 98th       | 0.020420                       | 2.00%                            |
| 99th       | 0.021131                       | 1.01%                            |


- **Fault Prediction**
  - Accuracy: **99.69%**
  - ROC-AUC: **0.9998**
  - Precision-Recall: **1.00 / 0.83**

## Why This Approach?
Unlike traditional end-to-end models, this modular approach allows **independent tuning** of anomaly detection and forecasting stages, improving interpretability and robustness. Moreover, the explanability of detected faults could be enhanced by separate analyses of reconstruction errors distributions.
