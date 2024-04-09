#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 19:21:44 2024

@author: Gi Won Kim
"""
import pandas as pd
from datetime import datetime

def calculate_age(birthday):
    """Calculates age in years given a birthday."""
    today = datetime.today()
    age = today.year - birthday.year - ((today.month, today.day) < (birthday.month, birthday.day))
    return age

# Function to determine the survey type
def get_survey_type(row):
    return "Baseline" if "Non-Disclose" in row.iloc[17] else "Informed"

#Function to convert sex into English 
def get_gender(row):
    return "Male" if "男" in row.iloc[18] else "Female"

# Function to extract the scores from a given row
def extract_scores(row, disclosure, score_columns):
    try:
        # Convert the 'Age' field from a birthday string to a datetime object
        birthday = pd.to_datetime(row.iloc[19])
        # Use the calculate_age function to get the age in years
        age = calculate_age(birthday) if pd.notnull(birthday) else None
       
        return {
            'Answer ID': row.iloc[0],
            'Sex': get_gender(row),
            'Age': age,
            'Survey Type': get_survey_type(row),
            'Content Source': score_columns[disclosure][5],
            'SME Size': score_columns[disclosure][6],
            'Content': score_columns[disclosure][8],
            'Product Type': score_columns[disclosure][7],
            'readability score': row.iloc[score_columns[disclosure][0]],
            'engagement score': row.iloc[score_columns[disclosure][1]],
            'trust score': row.iloc[score_columns[disclosure][2]],
            'purchase score': row.iloc[score_columns[disclosure][3]],
            'quality score': row.iloc[score_columns[disclosure][4]],
        }
    except IndexError as e:
        print(f"IndexError encountered for disclosure {disclosure} in row: {row.name}. Error: {e}")
        return None

# Function to process each row and create 6 new rows for the clean data
def process_row(row, score_columns):
    disclosures = row.iloc[17].split(',')[1:]  # Split and remove the first "分支块"
    clean_rows = [extract_scores(row, d, score_columns) for d in disclosures if d in score_columns]
    return [row for row in clean_rows if row is not None]

# Apply the processing function to each row
def process_data(cleaned_data, score_columns):
    processed_data = pd.DataFrame([score for _, row in cleaned_data.iterrows() for score in process_row(row, score_columns)])
    return processed_data
