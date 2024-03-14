from operator import index
#from pydantic_settings import BaseSettings
import streamlit as st
import plotly.express as px
import pandas_profiling
import pandas as pd
#from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import os 
import base64

def home():

    st.title(" STREAMLINED DATA PROFILING AND AUTOMATED MACHINE LEARNING MODEL GENERATION ")
    #st.image("https://www.challenge.org/wp-content/uploads/2019/03/gas-min-20-1.jpg",)
    
    file_ = open("img/home.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    _left, mid,a, _right = st.columns(4)
    with mid:
        st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
        )

    st.header("Introduction:")
    st.write("In the era of data-driven decision-making, the demand for streamlined and automated tools to analyze, interpret, and model datasets is more crucial than ever. This research project introduces an innovative solution that not only automates the data profiling process but also seamlessly integrates machine learning model creation, evaluation, and selection into a cohesive pipeline. Users can upload their datasets in CSV or Excel format, and the system will perform comprehensive profiling, preprocessing, and model evaluation, ultimately presenting the best-performing model for download.")
    st.write("A state-of-the-art method has been created to streamline and improve the entire data analysis and machine learning model selection process in the context of data-driven decision-making. It is easy for users to upload their datasets in CSV or Excel format, which initiates a thorough and automatic analysis in the profiling section. This step produces an extensive profile report that provides users with a concise, yet comprehensive, understanding of the features and organisation of their dataset. ")
    st.write("When it comes to modelling, users are free to select the kind of model that is needed, whether it is for a regression or classification task. The data is transformed into an 8:2 ratio for training and testing, and the system takes over seamlessly, automatically preprocessing features. This careful planning guarantees a reliable assessment of the model's performance on unknown data.")
    st.write("The next stage is to use the transformed datasets to simultaneously train multiple machine learning algorithms. A variety of metrics, including accuracy and root mean squared error (RMSE), are noted during the testing process for every model. In the section devoted to model comparison, these metrics function as quantitative indicators that enable a side-by-side comparison of the automatically generated models.")
    st.write("Using the metrics it has collected, the system displays a comparison table of all the models based on its performance. After that, users can download this excellent model from the easy-to-use download section and use it to easily integrate it into their own applications or analyse its outputs in more detail. Regardless of users' level of machine learning experience, the entire procedure is meant to provide them with a simple and useful tool for advanced data analysis and model selection.")

    st.header("Architecture")
    st.image("img\Arch.png")

    st.subheader("Project Owner:")
    st.text("Developed by: Vincent Isaac Jeyaraj J \nRegister No: 212220230060")

def upload():
    st.header("Upload Dataset")
    st.write("Upload The Dataset Only In The Accepted Format Other Formats Are Not Supported.")
    st.image("img/upload.png")

def profilling():
    st.header("Exploratory Data Analysis (EDA)")

    st.write("")
    st.image("img/eda2.png")
    
    st.subheader("EDA:")
    st.write("The method of studying and exploring record sets to apprehend their predominant traits, discover patterns, locate outliers, and identify relationships between variables. EDA is normally carried out as a preliminary step before undertaking extra formal statistical analyses or modeling.")
    st.write("Analyze's your dataset and provides a detailed Overview, Alerts, and Reproduction details on:")
    st.text("->Feature Stats \n->Feature Plots/Graphs \n->Correlations \n->Sample Data")

def modelling():
    st.header("Build Machine Leaarning Models & Compare")
    st.write("")

    st.subheader("Regression Model:")
    st.write("Correlations between dependent and independent variables are found using regression. Regression algorithms therefore aid in the prediction of continuous variables like housing prices, market trends, weather patterns, and the price of petrol and oil (a crucial task in today's world!") 
    st.write("The goal of the regression algorithm is to identify the mapping function that will allow us to map the continuous output variable, y, to the input variable, x.")

    st.subheader("Classification Model:")
    st.write("An algorithm called classification looks for functions that can assist in classifying the dataset according to different factors. When applying a classification algorithm, the data is categorised into different groups based on what the computer programme has learned after being trained on the training dataset.")
    st.write("The mapping function that converts the x input to the discrete y output is found by classification algorithms. Based on a specific set of independent variables, the algorithms estimate discrete values, or binary values, such as 0 and 1, yes and no, true or false. In simpler terms, classification algorithms work by fitting data to a logit function in order to predict the probability of an event occurring.")
    st.subheader("------------------------------------------------------------------------------------")

def performance(model_name):
    if os.path.exists('Classification.pkl'): 
        
        #st.subheader("Perfomance Metrics")
        
        st.subheader("Confusion Matrix:")
        st.write("""A confusion matrix is a matrix that summarizes the performance of a machine learning model on a set of test data. It is a means of displaying number of accurate and inaccurate instances from the model’s predictions. 
                 It is often used to measure the performance of classification models, which aim to predict a categorical label for each input instance. The matrix displays, the number of instances produced by the model on the test data.
                \n-> True positives (TP): occurs when the model accurately predicts a positive data point.
                \n-> True negatives (TN): occurs when the model accurately predicts a negative data point.
                \n-> False positives (FP): occurs when the model predicts a positive data point incorrectly.
                \n-> False negatives (FN): occurs when the model predicts a negative data point incorrectly.""")
        st.image("Confusion Matrix.png",width=600)

        st.subheader("AUC - ROC Curve:")
        st.write("""The AUC-ROC curve, or Area Under the Receiver Operating Characteristic curve, is a graphical representation of the performance of a binary classification model at various classification thresholds. 
                 It is commonly used in machine learning to assess the ability of a model to distinguish between two classes, typically the positive class (e.g., presence of a disease) and the negative class (e.g., absence of a disease). Let’s first understand the meaning of the two terms ROC and AUC.
                ->ROC: Receiver Operating Characteristics")
                ->AUC: Area Under Curve""")
        st.image("AUC.png",width=600)

        with open('Classification.pkl', 'rb') as f: 
            st.download_button('Download Model', f, file_name=model_name+".pkl")

    else:

        #st.subheader("Perfomance Metrics")
        
        st.subheader("Feature Importance Plot:")
        st.write("""A Feature Importance Plot in machine learning visually displays the significance of different input features in influencing a model's predictions. It ranks features based on their contribution to the model's performance, helping identify key variables affecting outcomes. 
                 This aids in feature selection, model interpretation, and understanding the impact of each input on the target variable. Higher-ranked features are deemed more influential, guiding data-driven decisions and facilitating the creation of simpler, more efficient models. 
                 Feature Importance Plots are valuable tools for enhancing model transparency and ensuring the inclusion of relevant information in predictive tasks.""")
        st.image("Feature Importance.png",width=600)

        st.subheader("Residuals:")
        st.write("""In machine learning, residuals represent the differences between predicted values and actual observed values in a model. 
                 Residuals quantify the model's accuracy by revealing how much it deviates from making perfect predictions. A well-fitted model exhibits residuals that are randomly distributed and centered around zero. 
                 Systematic patterns or trends in residuals indicate areas where the model may need improvement. Analyzing residuals aids in diagnosing model performance, identifying outliers, and refining algorithms to enhance predictive capabilities, 
                 ultimately contributing to better-informed decision-making in various applications, such as regression analysis and predictive modeling.""")
        st.image("Residuals.png",width=600)

        with open('Regression.pkl', 'rb') as f: 
            st.download_button('Download Model', f, file_name=model_name+".pkl")


if os.path.exists('dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

if os.path.exists('performance.csv'): 
    performance_df = pd.read_csv('performance.csv', index_col=None)

st.set_page_config(layout='wide')

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoML")
    choice = st.radio("Navigation", ["Home","Upload","Profiling","Modelling", "Performance"])
    st.info("This project application helps you build and explore your data.")



if choice=="Home":
    home()



if choice == "Upload":
    
    upload()
    file = st.file_uploader("Upload Your Dataset")

    try:
        st.dataframe(df)
        if st.button("Delete Dataframe"):
            st.info("Dataset deleted")
            os.remove("dataset.csv")
    except:
        pass

    if file:
        st.info(file)
        try: 
            if file.name[-3:] == 'csv':
                df = pd.read_csv(file, index_col=None)
            elif file.name[-3:] == 'lsx':
                df = pd.read_excel(file, index_col=None)
            df.to_csv('dataset.csv', index=None)
            st.dataframe(df)
        except:
            st.error("Please Upload a CSV or Excel File, type not supported.")
        


if choice == "Profiling": 
    
    profilling()
    try:
        if st.button("Generate Report"):
            #profile_df = ProfileReport(df,title="Profiling Report")
            profile_df = df.profile_report()
            st_profile_report(profile_df)
            export=profile_df.to_html()
            st.download_button(label="Download Full Report", data=export, file_name='report.html')
    except:
        st.error("Dataset Not Found !!!")


if choice == "Modelling": 
    modelling()
    try:
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        target_type = st.selectbox('Select Problem Type',['Regression','Classification'],index=0)
        if target_type=="Classification":
            from automl.classification import *
        elif target_type=="Regression":
            from automl.regression import *
        if st.button('Run Modelling'): 
            setup(df, target=chosen_target)
            setup_df = pull()
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            compare_df.to_csv('performance.csv', index=None)
            st.dataframe(compare_df)
            
            if target_type=="Classification":
                
                plot_model(best_model, plot = 'auc',display_format='streamlit', save=True)
                plot_model(best_model, plot = 'confusion_matrix',display_format='streamlit', save=True)

            elif target_type=="Regression":

                plot_model(best_model, plot = 'residuals',display_format='streamlit', save=True)
                plot_model(best_model, plot="feature", display_format="streamlit", save=True)

            save_model(best_model, target_type)
    except:
        st.error("Dataset Invalid/Not Found")
        st.info("Please check the datset\n1. Check format type\n 2. Try Re-uploading\n 3. Dataset features are not convertable")


if choice == "Performance": 
    st.header("Download and Visualize The Best Performing ML Model")
    
    try:
        model_name = performance_df.at[0,'Model']
        st.subheader("")
        st.subheader("Best model: " + model_name)
        st.write("Performace comaprision table of all the trained ML models.")
        
        #if st.button("Check all model performance"):
        st.dataframe(performance_df)
        performance(model_name)

        if st.button("Clear All Files Cache"):
                
                os.remove("performance.csv")
                os.remove("dataset.csv")
                try:
                    os.remove("Classification.pkl")
                    os.remove("Confusion Matrix.png")
                    os.remove("AUC.png")
                except:
                    os.remove("Regression.pkl")
                    os.remove("Feature Importance.png")
                    os.remove("Residuals.png")
                
                st.info("Cache Cleared")
    except:
        st.error("No model found.")
