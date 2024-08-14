import pandas as pd
import matplotlib.pyplot as plt

def load_dataset(filename='Combined Data.csv'):
    """
    Load the mental health dataset from a CSV file.

    Args:
    filename (str): The name of the CSV file to load.

    Returns:
    pandas.DataFrame: The loaded dataset.
    """
    return pd.read_csv(filename)

def perform_eda(data):
    """
    Perform initial exploratory data analysis on the mental health dataset.

    Args:
    data (pandas.DataFrame): The dataset to analyze.

    Returns:
    dict: A dictionary containing various EDA results.
    """
    eda_results = {}

    # Basic information about the dataset
    eda_results['info'] = data.info()

    # Descriptive statistics
    eda_results['describe'] = data.describe()

    # Count of each mental health status
    status_counts = data['status'].value_counts()
    eda_results['status_counts'] = status_counts

    # Visualize the distribution of mental health statuses
    plt.figure(figsize=(10, 6))
    status_counts.plot(kind='bar')
    plt.title('Distribution of Mental Health Statuses')
    plt.xlabel('Mental Health Status')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('mental_health_distribution.png')
    plt.close()

    # Calculate descriptive statistics using pandas
    eda_results['custom_stats'] = data['status'].describe()

    return eda_results

def analyze_sentiment_length(data):
    """
    Analyze the relationship between statement length and mental health status.

    Args:
    data (pandas.DataFrame): The dataset to analyze.

    Returns:
    dict: A dictionary containing analysis results.
    """
    analysis_results = {}

    # Calculate statement lengths
    data['statement_length'] = data['statement'].str.len()

    # Group by mental health status and calculate mean statement length
    mean_lengths = data.groupby('status')['statement_length'].mean().sort_values(ascending=False)
    analysis_results['mean_lengths'] = mean_lengths

    # Visualize mean statement lengths
    plt.figure(figsize=(10, 6))
    mean_lengths.plot(kind='bar')
    plt.title('Average Statement Length by Mental Health Status')
    plt.xlabel('Mental Health Status')
    plt.ylabel('Average Statement Length')
    plt.tight_layout()
    plt.savefig('statement_length_by_status.png')
    plt.close()

    return analysis_results

if __name__ == "__main__":
    # Load the dataset
    data = load_dataset()

    # Perform EDA
    eda_results = perform_eda(data)

    # Analyze sentiment length
    sentiment_length_results = analyze_sentiment_length(data)

    print("EDA completed. Results saved in 'eda_results' dictionary.")
    print("Sentiment length analysis completed. Results saved in 'sentiment_length_results' dictionary.")
    print("Visualizations saved as 'mental_health_distribution.png' and 'statement_length_by_status.png'.")
