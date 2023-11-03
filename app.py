import streamlit as st
import pandas as pd
from transformers import pipeline
from functions import pipe, zero_shot_classification, store_new_column


# Set page settings
st.set_page_config(
    page_title='Zero-Shot-Classification App',
    page_icon=None,
    layout="wide")


# Title
st.title('Zero-Shot Classification App')
st.write('---')


# Upload button
data = st.file_uploader(
    'Please, upload your dataset',
    type=['csv', 'xlsx', 'xls']
)


# Read the data as pandas DataFrame
if data is not None:
    
    st.write('### Data Preview')
    
    if data.name.endswith('.csv'):
        df = pd.read_csv(data)
    
    elif data.name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(data)



if data:

    st.dataframe(df)
    # Create a dropdown that contains the columns
    select_column = st.selectbox(
        'Select a column that contains the values you want to classify.',
        options = list(df.columns),
        index = None,
        placeholder = 'Choose an option'
    )
    # Create a text field for the labels
    input_labels = st.text_input(
        label='Please, select your labels.',
        placeholder='ex. Business Analyst, Data Analyst, Data Science, Customer Service'
    )

    
    
    if select_column and input_labels:
        # Perform the classification and get the list output to a variable.
        classified_values = zero_shot_classification(data=df[select_column], labels=input_labels)
        # Get the result
        st.write(classified_values)
        # Get success message
        st.success('Process Completed!')
        # Create a text field that gets a column name for your new column
        add_column_name = st.text_input(
            'Please, select a name for your new column.'
        )
        #
        store_button = st.button('Add Column')

        
        
    if store_button:
        # Save the classification values in a new column 
        store_new_column(data=df, column_name=add_column_name, list=classified_values)
        # Get success message
        st.success('Column has stored!')
        # Create a download button to download the updated dataset (with the new column)
        download_new_dataset_button = st.download_button(
            label='Download Dataset',
            data=df.to_csv(index=False),
            file_name='zero_shot_classification_dataset.csv')
            

