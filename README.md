# STREAMLINED DATA PROFILING AND AUTOMATED MACHINE LEARNING MODEL GENERATION

## Abstract
In a time when there is an abundance of data but little machine learning expertise, creating reliable predictive models is a major challenge. Our suggestion to tackle this problem is to put in place an Automated Machine Learning (AutoML) system with an intuitive web interface. With the goal of facilitating users with varying backgrounds and levels of experience, this system seeks to simplify the training and extraction of predictive models with ease, accuracy, and speed.

The four main parts of the project are model training, model evaluation, data profiling, and data upload. The process of developing a model can be started by users uploading datasets in structured formats like Excel or CSV with ease. Before moving on, users can better grasp their data by using interactive statistics and visualizations in the data profiling section, which offer insights into the dataset's features.

Users have the option to designate the model type (classification or regression) and target output feature after conducting data exploration. The AutoML system uses sophisticated algorithms to optimize model performance as it automatically trains a number of machine-learning models on the dataset. This allows for well-informed decision-making during the model selection process by offering detailed insights into model parameters and performance metrics.

To thoroughly evaluate model performance and make defensible decisions, users can access evaluation graphs and visualizations. Users are empowered to effectively leverage their data for a variety of applications by having the option to download the trained machine learning model after choosing the most accurate model for their particular needs.

This project alleviates the pressing need for affordable and effective solutions in the field of data analytics by democratizing the development of machine learning models. Data-driven decision-making and innovation are encouraged as it empowers them to leverage their data without requiring significant manual intervention.

## Architecture Diagram
![Sys_Architecture drawio (1)](https://github.com/vincent-isaac/Project-2/assets/75234588/15e73dcf-2390-4da2-a520-5fc29e80e926)

## Features
- User-friendly web interface for easy interaction.
- Automated data profiling to understand dataset characteristics.
- Automated model training with multiple algorithms.
- Hyperparameter optimization for optimal model performance.
- Model evaluation and comparison with visualizations.
- Downloadable trained machine learning models for deployment.

## Flow Diagram
![Flow_ML (1)](https://github.com/vincent-isaac/Project-2/assets/75234588/c423ee22-9f84-44c4-b047-775c52c94e82)


## Implementation
The project is implemented using the following technologies:
- Python: For backend development and machine learning algorithms.
- Streaamlit: For building the web application framework.
- scikit-learn, TensorFlow, Pandas: For machine learning algorithms and model training.

## Usage
To use the system, follow these steps:
1. Clone the repository to your local machine.
2. Install the necessary dependencies using `pip install -r requirements.txt`.
3. Run the streamlit web application using `streamlit run app.py`.
4. Access the web interface through your browser and upload your dataset.
5. Explore data characteristics, train multiple models, and evaluate their performance.
6. Download the trained model for deployment in production environments.

## Future Scope
- Improve model interpretability and explainability.
- Integrate Neural Networks and Transfer learning compatability.
- Expand functionality to support more advanced data analysis tasks.
- Implement continuous integration and deployment for automated testing and updates.

## Developed By
- [Vincent Isaac Jeyaraj J](https://github.com/vincent-isaac): Data Scientist & Researcher
