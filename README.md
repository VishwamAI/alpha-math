# Mathematics Dataset

This dataset code generates mathematical question and answer pairs, from a range
of question types at roughly school-level difficulty. This is designed to test
the mathematical learning and algebraic reasoning skills of learning models.

Original paper: [Analysing Mathematical
Reasoning Abilities of Neural Models](https://openreview.net/pdf?id=H1gR5iR5FX)
(Saxton, Grefenstette, Hill, Kohli).

## Example questions

```
Question: Solve -42*r + 27*c = -1167 and 130*r + 4*c = 372 for r.
Answer: 4

Question: Calculate -841880142.544 + 411127.
Answer: -841469015.544

Question: Let x(g) = 9*g + 1. Let q(c) = 2*c + 1. Let f(i) = 3*i - 39. Let w(j) = q(x(j)). Calculate f(w(a)).
Answer: 54*a - 30

Question: Let e(l) = l - 6. Is 2 a factor of both e(9) and 2?
Answer: False

Question: Let u(n) = -n**3 - n**2. Let e(c) = -2*c**3 + c. Let l(j) = -118*e(j) + 54*u(j). What is the derivative of l(a)?
Answer: 546*a**2 - 108*a - 118

Question: Three letters picked without replacement from qqqkkklkqkkk. Give prob of sequence qql.
Answer: 1/110
```

## Getting the source

### PyPI

The easiest way to get the source is to use pip:

```shell
$ pip install alpha-math
```

### From GitHub

Alternately you can get the source by cloning the mathematics_dataset
repository:

```shell
$ git clone https://github.com/deepmind/mathematics_dataset
$ pip install --upgrade mathematics_dataset/
```

## Generating examples

Generated examples can be printed to stdout via the `generate` script. For
example:

```shell
python -m mathematics_dataset.generate --filter=linear_1d
```

will generate example (question, answer) pairs for solving linear equations in
one variable.

We've also included `generate_to_file.py` as an example of how to write the
generated examples to text files. You can use this directly, or adapt it for
your generation and training needs.

## Dataset Metadata
The following table is necessary for this dataset to be indexed by search
engines such as <a href="https://g.co/datasetsearch">Google Dataset Search</a>.
<div itemscope itemtype="http://schema.org/Dataset">
<table>
  <tr>
    <th>property</th>
    <th>value</th>
  </tr>
  <tr>
    <td>name</td>
    <td><code itemprop="name">Mathematics Dataset</code></td>
  </tr>
  <tr>
  <tr>
    <td>description</td>
    <td><code itemprop="description">This dataset consists of mathematical question and answer pairs, from a range
of question types at roughly school-level difficulty. This is designed to test
the mathematical learning and algebraic reasoning skills of learning models.\n
\n
## Example questions\n
\n
```\n
Question: Solve -42*r + 27*c = -1167 and 130*r + 4*c = 372 for r.\n
Answer: 4\n
\n
Question: Calculate -841880142.544 + 411127.\n
Answer: -841469015.544\n
\n
Question: Let x(g) = 9*g + 1. Let q(c) = 2*c + 1. Let f(i) = 3*i - 39. Let w(j) = q(x(j)). Calculate f(w(a)).\n
Answer: 54*a - 30\n
```\n
\n
It contains 2 million
(question, answer) pairs per module, with questions limited to 160 characters in
length, and answers to 30 characters in length. Note the training data for each
question type is split into "train-easy", "train-medium", and "train-hard". This
allows training models via a curriculum. The data can also be mixed together
uniformly from these training datasets to obtain the results reported in the
paper. Categories:\n
\n
* **algebra** (linear equations, polynomial roots, sequences)\n
* **arithmetic** (pairwise operations and mixed expressions, surds)\n
* **calculus** (differentiation)\n
* **comparison** (closest numbers, pairwise comparisons, sorting)\n
* **measurement** (conversion, working with time)\n
* **numbers** (base conversion, remainders, common divisors and multiples,\n
  primality, place value, rounding numbers)\n
* **polynomials** (addition, simplification, composition, evaluating, expansion)\n
* **probability** (sampling without replacement)</code></td>
  </tr>
  <tr>
    <td>provider</td>
    <td>
      <div itemscope itemtype="http://schema.org/Organization" itemprop="provider">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">DeepMind</code></td>
          </tr>
          <tr>
            <td>sameAs</td>
            <td><code itemprop="sameAs">https://en.wikipedia.org/wiki/DeepMind</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
  <tr>
    <td>citation</td>
    <td><code itemprop="citation">https://identifiers.org/arxiv:1904.01557</code></td>
  </tr>
</table>
</div>

## Real-World Problem Solving

The `real_world_problems` module extends the capabilities of alpha-math to tackle real-world challenges. This module currently includes a mental health analysis component, demonstrating the application of data analysis and visualization techniques to a practical domain.

### Mental Health Dataset

The mental health dataset used in this module contains information about individuals' mental health statuses and their corresponding statements. This dataset allows us to explore the relationship between linguistic features and mental health conditions.

### Functions in mental_health_analysis.py

1. `load_dataset(filename='Combined Data.csv')`: Loads the mental health dataset from a CSV file.
2. `perform_eda(data)`: Conducts exploratory data analysis on the dataset, including basic information, descriptive statistics, and visualizations of mental health status distributions.
3. `analyze_sentiment_length(data)`: Analyzes the relationship between statement length and mental health status, providing insights into how the length of expressions might correlate with different mental health conditions.

### Running the Script

To run the mental health analysis script, use the following command:

```shell
python -m alphamath.real_world_problems.mental_health_analysis
```

### Interpreting the Results

The script generates two main outputs:

1. Exploratory Data Analysis (EDA) results: Stored in the `eda_results` dictionary, providing an overview of the dataset's characteristics.
2. Sentiment Length Analysis results: Stored in the `sentiment_length_results` dictionary, offering insights into the relationship between statement length and mental health status.

Additionally, two visualizations are generated:

1. `mental_health_distribution.png`: A bar chart showing the distribution of mental health statuses in the dataset.
2. `statement_length_by_status.png`: A bar chart displaying the average statement length for each mental health status.

These results and visualizations can be used to gain insights into the dataset and potentially inform further analysis or machine learning tasks related to mental health assessment based on textual data.
