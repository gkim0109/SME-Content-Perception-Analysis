#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 19:34:52 2024

@author: katy
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from numpy import sqrt
from scipy.stats import ttest_ind, f_oneway
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian
import statsmodels.formula.api as smf

def descriptive_stats_by_gender(clean_data):
    # Group the data by 'Survey Type' and 'Gender' and calculate descriptive statistics
    grouped_stats = clean_data.groupby(['Content Source', 'Sex']).agg({
        'readability score': ['count', 'mean', 'std'],
        'engagement score': ['mean', 'std'],
        'trust score': ['mean', 'std'],
        'purchase score': ['mean', 'std'],
        'quality score': ['mean', 'std']
    })
    
    # Flatten MultiIndex columns for clarity
    grouped_stats.columns = [' '.join(col).strip() for col in grouped_stats.columns.values]
    grouped_stats.reset_index(inplace=True)
    return grouped_stats

def plot_scores_by_content_source_gender(clean_data):
    """
    Plot consolidated bar charts for each score type by content source and gender, with data labels.
    """
    sns.set(style="whitegrid")
    score_types = ['readability score', 'engagement score', 'trust score', 'purchase score', 'quality score']
    
    # Create a figure with subplots in 2 rows
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharey=True)  # Adjust figsize as needed
    axes_list = [item for sublist in axes for item in sublist]  # Flatten the 2D list

    for i, score_type in enumerate(score_types):
        ax = axes_list[i]
        barplot = sns.barplot(
            x='Content Source', 
            y=f'{score_type} mean', 
            hue='Sex', 
            data=clean_data, 
            ax=ax,
            palette={'Male': '#56B4E9', 'Female': '#D07D3C'},  # Set custom colors
            capsize=.2,  # Increase capsize for visibility
            errorbar=('se')  # Use standard error instead of CI
        )
        ax.set_title(score_type.capitalize().replace('_', ' '), fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Score')
        ax.legend().set_visible(False)  # We'll create a custom legend later
        
        # Increase the font size for data labels for better visibility
        for p in barplot.patches:
            ax.text(p.get_x() + p.get_width() / 2., p.get_height(), 
                    f'{p.get_height():.2f}', 
                    ha="center", va='bottom', fontsize=14, color='black')  # Increased font size and added bold

    # Remove the empty subplot (if any)
    if len(score_types) % 2 != 0:
        fig.delaxes(axes_list[-1])

    # Create custom legend
    male_patch = mpatches.Patch(color='#56B4E9', label='Male')
    female_patch = mpatches.Patch(color='#D07D3C', label='Female')
    se_patch = mpatches.Patch(color='gray', label='SE for Error Bars')

    # Adjust the legend to show gender and SE info
    fig.legend(handles=[male_patch, female_patch, se_patch], loc='lower right', title='Legend')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the legend and main title
    plt.show()

def perform_grouped_analysis(clean_data):
    """
    Group data by 'Survey Type' and 'Content Source' to calculate means and confidence intervals.
    """
    grouped_data = clean_data.groupby(['Survey Type', 'Content Source']).agg({
        'readability score': ['mean', lambda x: stats.sem(x, ddof=0) * 1.96],
        'engagement score': ['mean', lambda x: stats.sem(x, ddof=0) * 1.96],
        'trust score': ['mean', lambda x: stats.sem(x, ddof=0) * 1.96],
        'purchase score': ['mean', lambda x: stats.sem(x, ddof=0) * 1.96],
        'quality score': ['mean', lambda x: stats.sem(x, ddof=0) * 1.96],
    }).reset_index()

    # Flatten MultiIndex columns for clarity
    grouped_data.columns = ['Survey Type', 'Content Source', 'Readability Mean', 'Readability CI', 'Engagement Mean', 'Engagement CI', 'Trust Mean', 'Trust CI', 'Purchase Mean', 'Purchase CI', 'Quality Mean', 'Quality CI']
    return grouped_data

def visualize_grouped_data(grouped_data):
    sns.set(style="whitegrid")
    plt.figure(figsize=(15, 10))

    score_types = ['Readability', 'Engagement', 'Trust', 'Purchase', 'Quality']
    for i, score_type in enumerate(score_types):
        ax = plt.subplot(3, 2, i + 1)
        barplot = sns.barplot(
            x='Survey Type',
            y=f'{score_type} Mean',
            hue='Content Source',
            data=grouped_data,
            ax=ax,
            capsize=.05
        )
        # Set bold title
        ax.set_title(f'{score_type} Scores',fontsize=14, fontweight='bold')
        # Set y-label
        ax.set_ylabel('Score')
        # Remove x-label
        ax.set_xlabel('')
        # Add data labels with two decimal places
        for p in barplot.patches:
            ax.text(
                p.get_x() + p.get_width() / 2., 
                p.get_height(), 
                f'{p.get_height():.2f}', 
                ha='center', 
                va='bottom',
                fontsize=14
            )

    # Adjust the layout
    plt.tight_layout()

    # Move the legend to the bottom right corner
    handles, labels = ax.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower right', title='Content Source', ncol=1)

    # Hide legends from subplots
    for ax in plt.gcf().axes:
        ax.get_legend().remove()

    plt.show()
    
# Function to calculate the mean scores and conduct T-tests and ANOVA for AI and Human content
def calculate_condition_means_and_tests(processed_data, survey_type):
    """
    Calculate mean scores for AI and Human content for a specific survey type ('Baseline' or 'Informed')
    and perform T-tests and ANOVA. Returns a DataFrame.
    
    Parameters:
    - processed_data (pd.DataFrame): DataFrame containing processed survey data.
    - survey_type (str): 'Baseline' or 'Informed'.
    
    Returns:
    - pd.DataFrame: DataFrame with mean scores, T-test, and ANOVA results.
    """
    # Filter data for the specified survey type
    condition_data = processed_data[processed_data['Survey Type'] == survey_type]

    # Initialize a list to store results
    results = []

    # Define score types for analysis
    score_types = ['readability score', 'engagement score', 'trust score', 'purchase score', 'quality score']

    # Loop over score types to calculate statistics
    for score in score_types:
        # Separate AI and Human content data
        ai_data = condition_data[condition_data['Content Source'] == 'AI'][score].dropna()
        human_data = condition_data[condition_data['Content Source'] == 'Human'][score].dropna()

        # Calculate mean scores
        ai_mean = ai_data.mean()
        human_mean = human_data.mean()

        # Perform T-test
        t_stat, p_val = ttest_ind(ai_data, human_data)
        
        # Perform ANOVA
        f_stat, p_val_anova = f_oneway(ai_data, human_data)

        # Append results
        results.append({
            'Score Type': score,
            'AI Mean': ai_mean,
            'Human Mean': human_mean,
            'T-Statistic': t_stat,
            'T-Test P-Value': p_val,
            'ANOVA F-Statistic': f_stat,
            'ANOVA P-Value': p_val_anova
        })

    # Convert list of results to DataFrame
    results_df = pd.DataFrame(results)

    return results_df

def perform_t_tests(clean_data):
    """
    Perform t-tests for each score type between Human and AI within each survey type.
    """
    results = []

    for survey_type in clean_data['Survey Type'].unique():
        for content_source in clean_data['Content Source'].unique():
            data_source = clean_data[(clean_data['Survey Type'] == survey_type) & (clean_data['Content Source'] == content_source)]
            for score_type in ['readability score', 'engagement score', 'trust score', 'purchase score', 'quality score']:
                scores = data_source[score_type.replace(' ', '_') + ' mean']  # Adjust field names as necessary
                mean_score = scores.mean()
                ci = stats.sem(scores) * 1.96  # Confidence interval
                results.append({
                    'Survey Type': survey_type,
                    'Content Source': content_source,
                    'Score Type': score_type,
                    'Mean Score': mean_score,
                    '95% CI': ci,
                })
    
    results_df = pd.DataFrame(results)
    return results_df

def comparison_analysis(clean_data):
    comparison_results = []

    # Define survey types and content sources for iteration
    survey_types = ['Baseline', 'Informed']
    content_sources = ['Human', 'AI']

    # Iterate over each combination of survey type and content source
    for survey_type in survey_types:
        for content_source in content_sources:
            # Filter data for current survey type and content source
            data_filtered = clean_data[(clean_data['Survey Type'] == survey_type) & (clean_data['Content Source'] == content_source)]
            
            # Calculate average scores for all score types
            for score_type in ['readability score', 'engagement score', 'trust score', 'purchase score', 'quality score']:
                avg_score = data_filtered[score_type].mean()
                
                # Store results
                comparison_results.append({
                    'Survey Type': survey_type,
                    'Content Source': content_source,
                    'Score Type': score_type,
                    'Average Score': avg_score
                })
    
    # Convert results to DataFrame
    comparison_df = pd.DataFrame(comparison_results)

    # Perform statistical tests for each score type between Human and AI within each survey type
    for survey_type in survey_types:
        for score_type in ['readability score', 'engagement score', 'trust score', 'purchase score', 'quality score']:
            human_scores = clean_data[(clean_data['Survey Type'] == survey_type) & (clean_data['Content Source'] == 'Human')][score_type]
            ai_scores = clean_data[(clean_data['Survey Type'] == survey_type) & (clean_data['Content Source'] == 'AI')][score_type]
            
            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(human_scores, ai_scores, nan_policy='omit')
            
            # Calculate Cohen's d
            cohen_d = (human_scores.mean() - ai_scores.mean()) / sqrt((human_scores.var() + ai_scores.var()) / 2)
            
            # Add comparison metrics to DataFrame
            comparison_df.loc[(comparison_df['Survey Type'] == survey_type) & (comparison_df['Score Type'] == score_type), 't-stat'] = t_stat
            comparison_df.loc[(comparison_df['Survey Type'] == survey_type) & (comparison_df['Score Type'] == score_type), 'p-value'] = p_value
            comparison_df.loc[(comparison_df['Survey Type'] == survey_type) & (comparison_df['Score Type'] == score_type), 'Cohen\'s d'] = cohen_d

    return comparison_df

def plot_comparison_analysis(comparison_data):
    sns.set(style="whitegrid")
    
    # Create a new column that combines 'Survey Type' and 'Content Source'
    comparison_data['Survey_Type_Content_Source'] = comparison_data['Content Source'] + ' ' + comparison_data['Survey Type']
    
    # Set the desired order for the bars
    desired_order = ['AI Baseline', 'Human Baseline', 'AI Informed', 'Human Informed']
    comparison_data['Survey_Type_Content_Source'] = pd.Categorical(
        comparison_data['Survey_Type_Content_Source'],
        categories=desired_order,
        ordered=True
    )

    #Define the size of the figure
    plt.figure(figsize=(14, 8))
    
    #Define a custom color palette
    custom_palette = {'AI Baseline': 'skyblue', 'AI Informed': 'steelblue',
                       'Human Baseline': 'sandybrown', 'Human Informed': 'chocolate'}
    
    # Create the barplot with the custom palette
    sns.barplot(
        x='Score Type',
        y='Average Score',
        hue='Survey_Type_Content_Source',
        data=comparison_data,
        palette= custom_palette
    )
    
    # Rotate x-axis labels if they overlap
    plt.xticks(rotation=45)
    
    # Add labels and title
    plt.ylabel('Average Score', fontweight='bold')
    plt.xlabel('Score Type', fontweight='bold')
    plt.title('Average Consumer Perception Scores by Survey Type and Content Source', fontweight='bold')
    
    # Add data labels on each bar if desired
    for p in plt.gca().patches:
        plt.text(p.get_x() + p.get_width() / 2., p.get_height(), f'{p.get_height():.2f}',
                 ha='center', va='bottom', fontsize=9)

    # Move the legend to the bottom right to avoid overlap
    plt.legend(loc='lower right')

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()

#SME Dimension Analysis
def sme_dimension_descriptive_analysis(processed_data, analysis_type):
    # Define score types for analysis
    
    score_types = ['readability score', 'engagement score', 'trust score', 'purchase score', 'quality score']

    # Group data by both SME Size and Content Source
    grouped_data = processed_data.groupby([analysis_type, 'Content Source'])

    # Calculate descriptive statistics
    descriptive_stats = grouped_data[score_types].agg(['mean', 'std']).reset_index()

    # Flatten MultiIndex columns for clarity
    descriptive_stats.columns = [' '.join(col).strip() for col in descriptive_stats.columns.values]

    # Return both descriptive statistics and T-test results
    return descriptive_stats

def plot_sme_descriptive_analysis(descriptive_stats):
    # Define the score types
    score_types = ['readability score', 'engagement score', 'trust score', 'purchase score', 'quality score']
    
    # Initialize a DataFrame for plotting
    plot_data = pd.DataFrame()

    # Process data for plotting
    for score in score_types:
        mean_column_name = f'{score} mean'
        if mean_column_name in descriptive_stats.columns:
            temp_df = descriptive_stats[['SME Size', 'Content Source', mean_column_name]].copy()
            temp_df.rename(columns={mean_column_name: 'Mean Score'}, inplace=True)
            temp_df['Score Type'] = score.capitalize().replace(' score', '')
            plot_data = pd.concat([plot_data, temp_df], axis=0)

    # Prepare data for micro and small businesses
    micro_data = plot_data[plot_data['SME Size'] == 'Micro']
    small_data = plot_data[plot_data['SME Size'] == 'Small']

    # Plot setup
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    sns.set(style="whitegrid")

    # Function to add data labels
    def add_data_labels(ax):
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 10), 
                        textcoords = 'offset points')

    # Plot for Micro businesses
    sns.barplot(ax=axes[0], x='Score Type', y='Mean Score', hue='Content Source', data=micro_data, palette='muted')
    axes[0].set_title('Micro Business Scores', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Mean Score', fontweight='bold')
    axes[0].set_xlabel('Score Type', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    add_data_labels(axes[0])

    # Plot for Small businesses
    sns.barplot(ax=axes[1], x='Score Type', y='Mean Score', hue='Content Source', data=small_data, palette='muted')
    axes[1].set_title('Small Business Scores', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Score Type', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    add_data_labels(axes[1])

    # Adjust legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', title='Content Source')

    # Hide the redundant legend
    axes[0].get_legend().remove()
    axes[1].get_legend().remove()

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2) # Adjust bottom to make space for the legend
    plt.show()

def plot_sme_product_descriptive_analysis(descriptive_stats):
    # Define the score types
    score_types = ['readability score', 'engagement score', 'trust score', 'purchase score', 'quality score']
    
    # Initialize a DataFrame for plotting
    plot_data = pd.DataFrame()

    # Process data for plotting
    for score in score_types:
        mean_column_name = f'{score} mean'
        if mean_column_name in descriptive_stats.columns:
            temp_df = descriptive_stats[['Product Type', 'Content Source', mean_column_name]].copy()
            temp_df.rename(columns={mean_column_name: 'Mean Score'}, inplace=True)
            temp_df['Score Type'] = score.capitalize().replace(' score', '')
            plot_data = pd.concat([plot_data, temp_df], axis=0)

    # Prepare data for material and experiemental product
    material_data = plot_data[plot_data['Product Type'] == 'Material']
    experiimental_data = plot_data[plot_data['Product Type'] == 'Experimental']

    # Plot setup
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    sns.set(style="whitegrid")

    # Function to add data labels
    def add_data_labels(ax):
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 10), 
                        textcoords = 'offset points')

    # Plot for Material Product
    sns.barplot(ax=axes[0], x='Score Type', y='Mean Score', hue='Content Source', data=material_data, palette='muted')
    axes[0].set_title('Material Product Scores', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Mean Score', fontweight='bold')
    axes[0].set_xlabel('Score Type', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    add_data_labels(axes[0])

    # Plot for Experimental businesses
    sns.barplot(ax=axes[1], x='Score Type', y='Mean Score', hue='Content Source', data=experiimental_data, palette='muted')
    axes[1].set_title('Experimental Product Scores', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Score Type', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    add_data_labels(axes[1])

    # Adjust legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', title='Content Source')

    # Hide the redundant legend
    axes[0].get_legend().remove()
    axes[1].get_legend().remove()

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2) # Adjust bottom to make space for the legend
    plt.show()
    
def compare_sme_sizes_for_ai_content(processed_data):
    """
    Perform comparative analysis between small and micro size businesses for AI-generated content.
    
    Parameters:
    processed_data (pd.DataFrame): The DataFrame containing processed survey data.
    """
    # Filter data for AI-generated content
    ai_data = processed_data[processed_data['Content Source'] == 'AI']
    
    # Further filter data for small and micro-sized businesses
    micro_data = ai_data[ai_data['SME Size'] == 'Micro']
    small_data = ai_data[ai_data['SME Size'] == 'Small']
    
    # Define the score types to analyze
    score_types = ['readability score', 'engagement score', 'trust score', 'purchase score', 'quality score']
    
    # Store results
    results = []
    
    # Perform t-tests for each score type
    for score in score_types:
        t_stat, p_value = ttest_ind(micro_data[score].dropna(), small_data[score].dropna())
        results.append({
            'Score Type': score,
            'Micro Mean': micro_data[score].mean(),
            'Small Mean': small_data[score].mean(),
            'T-Statistic': t_stat,
            'P-Value': p_value
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return results_df

def perform_perspective_correlation_analysis(processed_data):
    # Select the columns with the score types
    score_types = ['readability score', 'engagement score', 'trust score', 'purchase score', 'quality score']
    scores_data = processed_data[score_types]
    
    # Calculate Pearson correlation coefficients between the score types
    correlation_matrix = scores_data.corr(method='pearson')
    
    # Return the correlation matrix DataFrame
    return correlation_matrix


def plot_correlation_heatmap(correlation_matrix):
    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # Generate a heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='Greys', fmt=".2f",
                linewidths=.5, cbar_kws={"shrink": .5})

    # Add title and labels for clarity
    plt.title('Correlation Analysis Heatmap')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)

    # Show the heatmap
    plt.show()
    
def perform_perspective_correlation_analysis_ai_human(processed_data):
    # Define the score types
    score_types = ['readability score', 'engagement score', 'trust score', 'purchase score', 'quality score']
    
    # Split the data based on content type
    ai_data = processed_data[processed_data['Content Source'] == 'AI'][score_types]
    human_data = processed_data[processed_data['Content Source'] == 'Human'][score_types]
    
    # Calculate Pearson correlation coefficients for AI and Human content types
    ai_correlation_matrix = ai_data.corr(method='pearson')
    human_correlation_matrix = human_data.corr(method='pearson')
    # Plotting the heatmaps side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    
    # AI-generated content heatmap
    sns.heatmap(ai_correlation_matrix, annot=True, fmt=".2f", cmap='Reds', ax=ax1)
    ax1.set_title('AI-Generated Content Correlation')
    
    # Human-created content heatmap
    sns.heatmap(human_correlation_matrix, annot=True, fmt=".2f", cmap='Blues', ax=ax2)
    ax2.set_title('Human-Created Content Correlation')
    
    plt.tight_layout()
    plt.show()
    
def perform_gee_analysis(processed_data):
    # List of dependent variables
    dependent_vars = ['readability score', 'engagement score', 'trust score', 'purchase score', 'quality score']
    
    # Initialize list to store results
    results_list = []
    
    # Add a subject identifier for the GEE analysis
    processed_data['subject_id'] = processed_data.groupby('Answer ID').ngroup()
    
    # Perform GEE analysis for each dependent variable
    for dependent_var in dependent_vars:
        formula = f'Q("{dependent_var}") ~ C(Q("Survey Type")) * C(Q("Content Source"))'
        
        model = GEE.from_formula(formula, groups=processed_data['subject_id'], 
                                 cov_struct=Exchangeable(), data=processed_data, family=Gaussian())
        result = model.fit()
        
        # Extract key results
        for i in range(1, len(result.params)):
            coef_name = result.params.index[i]
            coef = result.params[i]
            pvalue = result.pvalues[i]
            ci_lower, ci_upper = result.conf_int().loc[coef_name]
            
            # Append results to list
            results_list.append({
                'Dependent Variable': dependent_var,
                'Coefficient Name': coef_name,
                'Coefficient': coef,
                'P-value': pvalue,
                '95% CI Lower': ci_lower,
                '95% CI Upper': ci_upper
            })
    
    # Convert results list to DataFrame
    results_df = pd.DataFrame(results_list)
    
    return results_df

def perform_gee_sme_content_analysis(processed_data):
    # List of dependent variables
    dependent_vars = ['readability score', 'engagement score', 'trust score', 'purchase score', 'quality score']
    
    # Initialize an empty DataFrame to store results
    results_df = pd.DataFrame()
    
    # Assuming 'subject_id' column is already present in processed_data
    # If not, uncomment the following line to create a 'subject_id' for GEE
    # processed_data['subject_id'] = processed_data.groupby('Answer ID').ngroup()

    # Iterate over each dependent variable and perform GEE
    for dependent_var in dependent_vars:
        # Define the model formula
        formula = f'Q("{dependent_var}") ~ C(Q("SME Size")) * C(Q("Content Source"))'
        
        # Fit the GEE model
        model = GEE.from_formula(formula, groups=processed_data['subject_id'], cov_struct=Exchangeable(), data=processed_data, family=Gaussian())
        result = model.fit()
        
        # Create a temporary DataFrame to hold results
        temp_df = pd.DataFrame({
            'Dependent Variable': dependent_var,
            'Parameter': result.params.index,
            'Coefficient': result.params.values,
            'Std. Error': result.bse.values,
            'z': result.tvalues.values,
            'P>|z|': result.pvalues.values,
            '95% CI Lower': result.conf_int().iloc[:, 0],
            '95% CI Upper': result.conf_int().iloc[:, 1]
        })
        
        # Append to the main results DataFrame
        results_df = pd.concat([results_df, temp_df], ignore_index=True)
    
    return results_df