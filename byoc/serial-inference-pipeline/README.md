# Builing custom images and deploy models to a SageMaker Serial Inference Pipeline

An inference pipeline is a Amazon SageMaker model that is composed of a linear sequence of two to fifteen containers that process requests for inferences on data. You use an inference pipeline to define and deploy any combination of pretrained SageMaker built-in algorithms and your own custom algorithms packaged in Docker containers. You can use an inference pipeline to combine preprocessing, predictions, and post-processing data science tasks. Inference pipelines are fully managed.

**References:**

1. [Host models along with pre-processing logic as serial inference pipeline behind one endpoint](https://docs.aws.amazon.com/sagemaker/latest/dg/inference-pipelines.html)
2. [Run Real-time Predictions with an Inference Pipeline](https://docs.aws.amazon.com/sagemaker/latest/dg/inference-pipeline-real-time.html)

## Build separate Docker images for Preprocessing and Prediction

1. Refer to [featurizer](./featurizer/) for custom BYOC implemenation of preprocessor using Scikit-learn.
2. Refer to [xgbpredictor](./xgbpredictor/) for custom BYOC implemenation of XGBoost estimator
