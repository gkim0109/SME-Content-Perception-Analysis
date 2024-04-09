#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 19:21:44 2024

@author: katy
"""

import pandas as pd
from data_cleaning import load_and_clean_data
from data_conversion import process_data
from data_analysis import perform_grouped_analysis, visualize_grouped_data, perform_t_tests, comparison_analysis, descriptive_stats_by_gender, plot_scores_by_content_source_gender, calculate_condition_means_and_tests, plot_comparison_analysis, sme_dimension_descriptive_analysis, plot_sme_descriptive_analysis, compare_sme_sizes_for_ai_content, perform_perspective_correlation_analysis, plot_correlation_heatmap, perform_perspective_correlation_analysis_ai_human, plot_sme_product_descriptive_analysis, perform_gee_analysis, perform_gee_sme_content_analysis 

# Define the path to your survey data
file_path = '/Users/katy/Documents/PKU-GMBA-22/Course Materials/Academic Thesis/Thesis Analysis/Data/Crowdsource Data.xlsx'

# Load raw survey data to include in the output
raw_survey_data = pd.read_excel(file_path)

# Load and Clean Data
cleaned_data, removed_data = load_and_clean_data(file_path)

# Define the score columns for data conversion, ensure these are aligned with the actual data structure
score_columns = {
    'Non-Disclose1': (24, 25, 26, 27, 28, 'Human', 'Micro', 'Material' , 'Micro1'),
    'Non-Disclose2': (32, 33, 34, 35, 36, 'AI', 'Micro', 'Material', 'Micro1'),
    'Non-Disclose3': (40, 41, 42, 43, 44, 'Human', 'Micro', 'Experimental', 'Micro2'),
    'Non-Disclose4': (48, 49, 50, 51, 52, 'AI', 'Micro', 'Experimental', 'Micro2'),
    'Non-Disclose5': (56, 57, 58, 59, 60, 'Human', 'Small', 'Material', 'Small'),
    'Non-Disclose6': (64, 65, 66, 67, 68, 'AI', 'Small', 'Material', 'Small'),
    'Disclose1': (73, 74, 75, 76, 77, 'Human', 'Micro', 'Material', 'Micro1'),
    'Disclose2': (81, 82, 83, 84, 85, 'AI', 'Micro', 'Material', 'Micro1'),
    'Disclose3': (89, 90, 91, 92, 93, 'Human', 'Micro', 'Experimental', 'Micro2'),
    'Disclose4': (97, 98, 99, 100, 101, 'AI', 'Micro', 'Experimental', 'Micro2'),
    'Disclose5': (105, 106, 107, 108, 109, 'Human', 'Small', 'Material', 'Small'),
    'Disclose6': (113, 114, 115, 116, 117, 'AI', 'Small', 'Material', 'Small'),
}

# Convert Data
processed_data = process_data(cleaned_data, score_columns)

# Analyze Data
grouped_data = perform_grouped_analysis(processed_data)
visualize_grouped_data(grouped_data)
grouped_data_gender = descriptive_stats_by_gender(processed_data)
plot_scores_by_content_source_gender(grouped_data_gender)

#perform condition analysis 
baseline_analysis = calculate_condition_means_and_tests(processed_data, "Baseline")
informed_analysis = calculate_condition_means_and_tests(processed_data, "Informed")

#perform comparison analysis
comparison_results = comparison_analysis(processed_data)
plot_comparison_analysis(comparison_results)

#Perform SME Size Analysis
descriptive_stats_sme_size = sme_dimension_descriptive_analysis(processed_data, 'SME Size')
comparative_analysis_sme_size_ai = compare_sme_sizes_for_ai_content(processed_data)
plot_sme_descriptive_analysis(descriptive_stats_sme_size)

#Perform SME Product Analysis
descriptive_stats_sme_product_type = sme_dimension_descriptive_analysis(processed_data, 'Product Type')
plot_sme_product_descriptive_analysis(descriptive_stats_sme_product_type)

#perform consumer perspective correlation analysis
perspective_correlation_analysis = perform_perspective_correlation_analysis(processed_data)
plot_correlation_heatmap(perspective_correlation_analysis)
perform_perspective_correlation_analysis_ai_human(processed_data)

#perform GEE analysis 
gee_analysis = perform_gee_analysis(processed_data)
gee_analysis_by_sme_size = perform_gee_sme_content_analysis(processed_data)

# Write the final output data, including original, cleaned, processed, and analysis results
output_path = '/Users/katy/Documents/PKU-GMBA-22/Course Materials/Academic Thesis/Thesis Analysis/Output/Survey Data Analysis Results.xlsx'
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

    raw_survey_data.to_excel(writer, sheet_name='Raw Survey Data', index=False)
    cleaned_data.to_excel(writer, sheet_name='Cleaned Survey Data', index=False)
    removed_data.to_excel(writer, sheet_name='Removed Survey Data', index=False)
    processed_data.to_excel(writer, sheet_name='Processed Survey Data', index=False)
    grouped_data.to_excel(writer, sheet_name='Grouped Analysis', index=False)
    grouped_data_gender.to_excel(writer, sheet_name='Grouped Gender Analysis', index=False)
    baseline_analysis.to_excel(writer, sheet_name='Baseline Analysis', index=False)
    informed_analysis.to_excel(writer, sheet_name='Informed Analysis', index=False)
    comparison_results.to_excel(writer, sheet_name='Comparison Analysis', index=False)
    descriptive_stats_sme_size.to_excel(writer, sheet_name='SME Size Descriptive Analysis', index=False)
    comparative_analysis_sme_size_ai.to_excel(writer, sheet_name='SME Size Comparison Analysis', index=False)
    perspective_correlation_analysis.to_excel(writer, sheet_name='Correlation Analysis', index=False)
    descriptive_stats_sme_product_type.to_excel(writer, sheet_name='Product Type Analysis', index=False)
    gee_analysis.to_excel(writer, sheet_name='Gee Analysis', index=False)
    gee_analysis_by_sme_size.to_excel(writer, sheet_name='Gee Analysis - SME Size', index=False)


print("The data processing workflow has completed successfully and the output has been saved.")
