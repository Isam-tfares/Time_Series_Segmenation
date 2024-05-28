# Time Series Segmentation

Time series segmentation is a crucial project in the field of data analysis and signal processing. It involves dividing a continuous time series data set into discrete segments where the data within each segment is more homogeneous or has similar characteristics. For instance, consider a time series graph showing the hourly temperature of a city over a week. Segmenting this data could reveal patterns like daily temperature cycles or sudden changes due to weather fronts. This segmentation is essential for several reasons:

1. **Pattern Recognition and Anomaly Detection**: By breaking down the time series into segments, it becomes easier to identify patterns, trends, and anomalies. This is particularly important in fields like finance (for stock market analysis), meteorology (for weather pattern analysis), and healthcare (for monitoring patient vital signs).

2. **Simplifying Complex Data**: Time series data can be incredibly complex and voluminous. Segmenting the data simplifies the analysis by breaking it down into manageable parts, making it easier to understand and interpret.

3. **Improving Predictive Modeling**: In machine learning and statistical modeling, segmented time series data can improve the accuracy of predictive models. By training models on specific segments, the models can better understand and predict future data points within those segments.

4. **Efficiency in Data Processing**: Segmenting time series data can lead to more efficient data processing. Algorithms can focus on smaller, more relevant segments of data, reducing computational load and improving processing speed.

5. **Customized Analysis**: Different segments of time series data can exhibit different behaviors. By segmenting the data, analysts can apply customized approaches to each segment, leading to more nuanced and accurate analysis.

6. **Real-time Monitoring and Decision Making**: In real-time systems, such as network traffic monitoring or industrial process control, time series segmentation allows for immediate detection of changes in the system, facilitating prompt decision-making and action.

7. **Historical Analysis and Forecasting**: In historical data analysis, segmentation helps in identifying periods of change or stability, which is crucial for understanding past trends and forecasting future ones.

Overall, the time series segmentation project is vital for extracting meaningful information from sequential data, enabling better decision-making, and providing insights that would be difficult to obtain from raw, unsegmented data.

## Project Structure

This project is structured into several main folders:

### 1. Dataset
This folder contains various datasets used in the project, including:
- human_activity_segmentation_challenge-main dataset
- mobile-sensing-human-activity-data-set-main dataset
- Skoltech_Anomaly_Benchmark dataset
- Time_Series_Segmentation_Benchmark dataset
- Finding Turing_Change_Point_Dataset dataset

### 2. Benchmark
This folder contains the implementation of various algorithms and their evaluations:
- **Algorithms**: Implementations of several state-of-the-art segmentation algorithms, including:
  - BinaryClaSPSegmentation
  - DynamicProgramming
  - FLOSS
  - Pelt
  - BOCD
  - Fluss
  - Vanilla_LSTM
- **Evaluation**: Evaluations of the algorithms on the datasets using metrics such as f_measure, covering, and NAB.
  - **Results**: Contains the results of the evaluations.

## Algorithms

### BinaryClaSPSegmentation
lassification Score Profile (ClaSP) is a method for time series segmentation based on self-supervision. It involves partitioning the time series into overlapping windows of a fixed length and generating hypothetical splits. For each split, a binary classification problem is created, labeling windows to the left of the split as 0 and to the right as 1. A k-Nearest-Neighbor (k-NN) classifier is trained and evaluated, with the cross-validation score used to measure dissimilarity between windows. Local maxima in the classification score profile indicate potential change points.

### DynamicProgramming
Dynamic Programming for change point detection involves finding the optimal segmentation of a time series by minimizing a cost function. This approach uses dynamic programming to efficiently compute the minimum cost for segmenting the time series at different points.

### FLOSS
Fast Low-cost Online Semantic Segmentation (FLOSS) is an online algorithm for time series segmentation. It uses a sliding window approach to detect changes in the semantic structure of the time series.

### Pelt
Pruned Exact Linear Time (PELT) is a method for change point detection that minimizes the sum of the error within segments plus a penalty for each change point. PELT uses dynamic programming and pruning to achieve exact and efficient segmentation.

### BOCD
Bayesian Online Change Point Detection (BOCD) is a probabilistic method for detecting change points in a time series. It uses Bayesian inference to update the probability of a change point occurring at each time step.

### Fluss
Fast Low-cost Unipotent Semantic Segmentation (FLUSS) is similar to FLOSS but optimized for unipotent (single variable) time series. It detects semantic changes using a sliding window and low-cost computations.

### Vanilla_LSTM
Vanilla Long Short-Term Memory (LSTM) networks are a type of recurrent neural network (RNN) used for time series prediction and segmentation. LSTMs can capture long-term dependencies in time series data, making them suitable for detecting change points.

## Evaluation Metrics

The evaluation of the algorithms is based on the following metrics:
- **f_measure**: A measure of a test's accuracy.
- **covering**: A metric for evaluating the similarity between two segmentations.
- **NAB**: The Numenta Anomaly Benchmark, used for evaluating anomaly detection algorithms.

## Conclusion

Time series segmentation is a powerful tool for data analysis, providing insights and improving the efficiency and accuracy of predictive models. By leveraging various segmentation algorithms, we can better understand and analyze complex time series data.

## Future Work

Future work may include exploring more advanced machine learning models and techniques for time series segmentation, further optimization of current algorithms, and applying the developed methods to a wider range of datasets.
