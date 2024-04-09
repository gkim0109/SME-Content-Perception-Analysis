#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:34:08 2024

@author: Gi Won Kim
"""
import pandas as pd
from datetime import datetime

def calculate_age(birthdate, reference_date):
    """Calculate age given a birthdate and a reference date."""
    age = reference_date.year - birthdate.year - ((reference_date.month, reference_date.day) < (birthdate.month, birthdate.day))
    return age

def count_missed_attention_checks(row, condition, excel_columns, correct_answers_baseline, correct_answers_informed):
    """Count the number of missed attention checks for a given survey response."""
    if condition == 'baseline':
        indices = [excel_columns[col] for col in ['W', 'AE', 'AM', 'AU', 'BC', 'BK']]
        correct_answers = correct_answers_baseline
    else:  # 'informed'
        indices = [excel_columns[col] for col in ['BT', 'CB', 'CJ', 'CR', 'CZ', 'DH']]
        correct_answers = correct_answers_informed
    
    missed_checks = sum(1 for i, answer in zip(indices, correct_answers) if row[i] != answer)
    return missed_checks

def detect_straightlining(row, score_columns):
    """Detect straightlining behavior for a given survey response across specified score columns."""
    straightlining_detected = False
    for disclosure, columns in score_columns.items():
        # columns[0] to columns[4] represent the range of scores for a single post
        for start in range(columns[0]-1, columns[4], 5):  # Adjust for zero-based indexing and step through each post
            scores = row.iloc[start:start+5]  # Extract scores for each post
            if scores.nunique() == 1 and pd.notnull(scores).all():  # Check if all answers are the same and not null
                straightlining_detected = True
                break  # Break from the inner loop if straightlining is detected for any post
        if straightlining_detected:
            break  # Break from the outer loop if straightlining is detected
    return straightlining_detected

def load_and_clean_data(file_path, reference_date=datetime.now()):
    """Load survey data from an Excel file, clean it based on specified criteria, and separate removed responses."""
    # Load the data
    df = pd.read_excel(file_path, header=1)

    # Excel column indices for attention check questions
    excel_columns = {
        'W': 22, 'AE': 30, 'AM': 38, 'AU': 46, 'BC': 54, 'BK': 62,  # Baseline condition columns
        'BT': 71, 'CB': 79, 'CJ': 87, 'CR': 95, 'CZ': 103, 'DH': 111  # Informed condition columns
    }
    
    # Define the score columns for each type of disclosure
    score_columns = {
        'Non-Disclose1': (24, 25, 26, 27, 28, 'Human'),
        'Non-Disclose2': (32, 33, 34, 35, 36, 'AI'),
        'Non-Disclose3': (40, 41, 42, 43, 44, 'Human'),
        'Non-Disclose4': (48, 49, 50, 51, 52, 'AI'),
        'Non-Disclose5': (56, 57, 58, 59, 60, 'Human'),
        'Non-Disclose6': (64, 65, 66, 67, 68, 'AI'),
        'Disclose1': (73, 74, 75, 76, 77, 'Human'),
        'Disclose2': (81, 82, 83, 84, 85, 'AI'),
        'Disclose3': (89, 90, 91, 92, 93, 'Human'),
        'Disclose4': (97, 98, 99, 100, 101, 'AI'),
        'Disclose5': (105, 106, 107, 108, 109, 'Human'),
        'Disclose6': (113, 114, 115, 116, 117, 'AI'),
    }


    # Correct answers for attention check questions
    correct_answers_baseline = ['是', '眼镜袋', '是', '是', '塑料', '否']
    correct_answers_informed = ['是', '眼镜袋', '是', '是', '塑料', '是']

    # Determine the condition for each respondent
    df['Condition'] = df.iloc[:, 17].apply(lambda x: 'baseline' if 'Non-Disclose' in x else 'informed')

    # Count missed attention checks for each respondent
    df['Missed_Attention_Checks'] = df.apply(
        lambda row: count_missed_attention_checks(
            row, 
            row['Condition'], 
            excel_columns, 
            correct_answers_baseline, 
            correct_answers_informed
        ), 
        axis=1
    )

    # Apply the completion time condition
    completion_times = df.iloc[:, 4]  # Assuming completion times are in column 5 (zero-based index 4)
    mean_time = completion_times.mean()
    std_dev_time = completion_times.std()
    time_threshold = mean_time - std_dev_time
    df['Below_Time_Threshold'] = completion_times < time_threshold
    
    # Apply Age condition
    # Convert birthday column from string to datetime objects and calculate age
    df.iloc[:, 19] = pd.to_datetime(df.iloc[:, 19], errors='coerce')
    df['Age'] = df.iloc[:, 19].apply(lambda x: calculate_age(x, reference_date) if pd.notnull(x) else None)

    # Apply the age condition by flagging those not meeting the age threshold
    df['Age_Out_Of_Range'] = ~df['Age'].between(18, 35, inclusive='both') | df['Age'].isnull()

     # Apply straightlining detection
    df['Straightlining'] = df.apply(lambda row: detect_straightlining(row, score_columns), axis=1)

    # Final Filtering
    cleaned_df = df[(df['Missed_Attention_Checks'] <= 1) & (~df['Below_Time_Threshold']) & (~df['Straightlining']) & (~df['Age_Out_Of_Range'])]
    removed_responses_df = df[~df.index.isin(cleaned_df.index)]

    return cleaned_df, removed_responses_df
