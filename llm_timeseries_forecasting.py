# -*- coding: utf-8 -*-
"""LLM-timeseries-forecasting.ipynb




# LLMTime - Zero-shot prompting LLMs for time series forecasting

In this Project, I will explore the use of zero-shot prompting with Large Language Models (LLMs) for time series forecasting

we will leverage advanced pre-trained models like GPT-3, which offer powerful probabilistic tools, including likelihood evaluation and sampling. Remarkably, LLMs can be applied directly to time series data without any fine-tuning, enabling zero-shot learning. Additionally, LLMs can generate explanations for their predictions, enhancing our understanding of their outputs.

But why are LLMs effective for time series forecasting? Authors provide the answer in LLM's preference for simplicity (Occam's razor) - LLMs tend to favor simple or repetitive patterns. This aligns well with common time series features like seasonality.

However, applying LLMs to time series data presents unique challenges compared to traditional language modeling:

1. **Numerical Sequences**: Time series data consists of numerical values, not words.
2. **Complex Probability Distributions**: Language models excel at representing discrete distributions, whereas time series require continuous distributions.
3. **Tokenization Variability**: The representation of numbers can differ based on the tokenizer. For example, '4223560' might be tokenized as [422, 35, 630] using the GPT tokenizer. This variability can impact model performance. To address this, the LLaMA tokenizer, as highlighted by Touvron et al. in 2023, maps numbers to individual digits, significantly enhancing the model's mathematical capabilities.

Through this project ,my aim to explore potential of using LLMs in time series forecasting. We will use weather forecasting to experiment with this method.

Notable differences from a typical machine learning pipeline are as follows -
1. We don't need any sophisticated ML libraries such as scikit or PyTorch.
2. We don't need high performance computing infrastructure. CPUs will do just fine.
3. Since we will be querying OpenAI's API, it is crucial to think about the parameters to reduce the API costs.
4. Most of the work goes in forming the prompt and decoding the outputs as compared to thinking about modeling choices in ML.

### Objectives
In this tutorial, our aim to:
1. Acquaint you with the application of machine learning techniques using Large Language Models (LLMs).
2. Enhance your understanding of LLMs and the parameters influencing their behavior.
3. Guide you through the essentials for successful time series prediction with LLMs.
4. Translate knowledge from transformers to the realm of LLMs.

### Task: Time Series Analysis of Weather Data
- **Objective**: Predict the average maximum temperature (T_max) for the upcoming weeks. Our goal is to forecast average T_max up to 6 months (24 weeks) ahead.
- **Evaluation**: The predictions will be assessed using the Mean Absolute Error (MAE) metric.
- **Validation and Tuning**: We'll utilize Negative Log-Likelihood per Dimension (NLL/D) for hyperparameter tuning and validation.

#### Caveats: Mindful API Usage
Each API call incurs a cost. It's crucial to be mindful of the frequency of calls and the parameters used, as they can quickly add to the overall expense.

### Setup

Install the following libraries:

1. `tiktoken`: Library containing tokenizers for the majority of LLMs. This tutorial has been built using `tiktoken==0.5.1`
2. `openai`: Library to call OpenAI's API. We will be using GPT-3 mostly. This tutorial has been built using `openai==1.2.2`
3. `jax`: Library to compute gradient of a transformation. Only used once in the tutorial. This tutorial has been built using `jax==0.4.18`


**Setup OpenAI API access:** To set up your OpenAI API access and begin using it for your projects, follow these  steps:

1. **Create an OpenAI Account**: Visit the OpenAI Platform website at [platform.openai.com](https://platform.openai.com) and sign in or create a new account if you don't already have one.

2. **Generate an API Key**: After logging in, click your profile icon located at the top-right corner of the page. Select "View API Keys" from the dropdown menu. Then, you'll find an option to "Create New Secret Key". Click this button to generate your new API key. Save your API key immediately after generation, as you won't be able to view it again once the window showing it closes.

3. **Free Credits and Billing Information**: As a new user, you will receive $5 worth of free credit upon creating your API key. This credit expires after three months. After using up this credit or upon its expiration, you can enter your billing information to continue using the API. If you don't provide billing information, you will retain login access but won't be able to make further API requests.

4. **Storing Your API Key**: Store your API key securely. It's a common practice to put the key in a `.env` file within your project. For this tutorial, we can either provide it manually or put the key in environment variable by using `export OPEN_API_KEY=YOUR_KEY`. Please consult your tutor for the right way to access the API.

By following these steps, you can successfully set up and start using OpenAI's API for your projects. Remember to handle your API key with care, as it provides access to OpenAI's powerful AI capabilities.
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

import tiktoken
from openai import OpenAI
from jax import vmap, grad # Only used for computing log-likelihood

client = OpenAI(
#     api_key = "PUT YOUR KEY HERE"
    api_key = os.environ.get("OPENAI_API_KEY")
)

# information regarding specific LLMs
MODELS = {
    'GPT-3': {
        'model_name': 'text-davinci-003',
        'context_length': 4097
    }
}

# Other Parameters
LLM_MODEL = 'GPT-3'
MODEL_NAME = MODELS[LLM_MODEL]['model_name']
CONTEXT_LENGTH = MODELS[LLM_MODEL]['context_length']

"""## Time series data for weather forecasting

We are interested in weather forecasting. Data is provided in `data/` folder. The following code block loads the data for time series to be used for training and testing.

Please follow the steps to ensure you understand what's loaded in `y_train` and `y_test`.

As a benchmark, we also have `predictions.csv` file with predictions from seasonal arima and ground truth time series.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load in the dataset
df_test = pd.read_csv('data/weather_test_data.csv')
df_train = pd.read_csv('data/weather_train_data.csv')

# Look at the test and training data
print("Original Training/Testing dataframes: \n")
print(df_test.head())
print(df_train.head())
print('\n')

print(f'Size of training set: {len(df_train)}')
print(f'Size of test set: {len(df_test)}')
print('\n')

# Convert Date column to datetime type
df_test['Date'] = pd.to_datetime(df_test['Date'])
df_train['Date'] = pd.to_datetime(df_train['Date'])

# Resample by week and take the mean
df_train = df_train.resample('W', on='Date').mean()
df_test = df_test.resample('W', on='Date').mean()
print(f'Number of train observations, resampled by week: {len(df_train)}')
print(f'Number of test observations, resampled by week: {len(df_test)}')
print('\n')

y_train = df_train['tmax'].values
# t_train = df_train.index
# y_train = y_train.reshape((len(y_train), 1))

y_test = df_test['tmax'].values
# t_test = df_test.index
# y_test = y_test.reshape((len(y_test), 1))

print("Training time series (first 10 values):")
print(y_train[:10])
print('\n')

print("Testing time series (first 10 values):")
print(y_test[:10])

"""Now, look at the benchmark data and ensure that the `test` column contains the same values as processed from above. The additional columns are predictions from the other methods."""

# Read the `predcitions.csv` data in `benchmarks`

benchmarks = pd.read_csv('predictions.csv', index_col=0)

benchmarks.head()

"""We will follow these steps to establish the forecasts.

<img src='img/llmtime.png' width=750>

### Time Series Data Preparation for LLM Analysis

#### Step 1: Normalization:

Scale the numbers in the time series. The authors propose to utilize Min-Max scaling with a twist.

**Procedure**:
  - Allow \(\alpha\) percentile of numbers in the training time series to be below 1.
  - Optionally, offset the time series by \(\beta\) times the range of the series before scaling.
  - Choose \(\alpha\) and \(\beta\) as hyperparameters.

#### Step 2: Serializing
Standardize the decimal precision of numbers.

**Procedure**:
  - Set `prec` as the number of digits allowed after the decimal.
  - Note: The number of digits before the decimal is not fixed.

#### Step 3: Truncation/Pre-processing the Input
Adapt the time series for the LLM's input constraints.

**Procedure**:
  - Form a time series string from the processed numbers.
  - If the string exceeds the model's context limit, truncate it appropriately.
  - Iteratively determine the optimal input string length for the model.

Take sometime to go through the following classes and the `get_scaler` function to understand the scaling scheme.
"""

@dataclass
class SerializerSettings:
    """
    Settings for serialization of numbers.
    """
    prec: int = 3 # number of digits after the decimal
    base: int = 10
    signed: bool = True # Whether the inputs to the LLM indicate sign of the numbers.
    max_val: bool = 1e7 # This is used in deciding the number of digits before the decimal point
    time_sep: str = ' ,' # How to delimit each time step
    bit_sep: str = ' ' # How to separate each digit. It depends on the tokenizer.
    plus_sign: str = ''
    minus_sign: str = ' -'
    missing_str: str = ' Nan' # How to represent missing entries
    half_bin_correction: bool = True # To adapt the discrete distribution to continuos distribution, we do bin correction such that if the prediction is 0.12, corrected version will be 0.125.

@dataclass
class Scaler:
    """
    Represents a data scaler with transformation and inverse transformation functions.

    Attributes:
        transform (callable): Function to apply transformation.
        inv_transform (callable): Function to apply inverse transformation.
    """
    transform: callable = lambda x: x
    inv_transform: callable = lambda x: x

"""### Step 1: Normalization"""

def get_scaler(time_series: np.array, scaler_type: str = 'advanced', alpha: float = 0.95, beta: float = 0.3):
    """
    Generates a Sclaer object based on the values in time_series.

    Args:
        time_series: 1D array Data to base scaling on.
        scaler_type: Type of scaler to be used.
        alpha: Quantile for scaling.
        beta: Shift parameter.
    """
    if scaler_type == 'gaussian':
        mean = time_series.mean()
        std = time_series.std()

        def transform(x):
            return (x - mean)/std

        def inv_transform(x):
            return std * x + mean


    elif scaler_type == 'basic':
        # Time series is scaled by the alpha quantile of absolute values
        data = time_series[~np.isnan(time_series)]
        q = np.maximum(np.quantile(np.abs(data), alpha), 0.01)

        def transform(x):
            return x / q

        def inv_transform(x):
            return q * x

    elif scaler_type == 'advanced':
        # Time series is shifted by beta * range and scaled by the alpha quantile
        min_ = np.min(time_series) - beta*(np.max(time_series)-np.min(time_series))
        q = np.quantile(time_series-min_, alpha)
        if q == 0:
            q = 1

        def transform(x):
            return (x - min_) / q

        def inv_transform(x):
            return x * q + min_

    else:
        raise ValueError(f'Unrecognized `scaler_type`: {scaler_type}.')

    return Scaler(transform=transform, inv_transform=inv_transform)

# basic scaler should have alpha quantile as 1.0 (assuming all values are positive)
val = y_train
assert np.all(val > 0), "Some values are negative"
scaler = get_scaler(val, 'basic', alpha=0.90)
assert np.quantile(scaler.transform(val), 0.90) == 1.0

print(scaler.transform(val)[:10])

"""### Step 2: Serializing"""

def num2bits(val: np.array, settings: SerializerSettings):
    """
    Converts each value in a time series to it's bit representation.

    Args:
        val: Series to be converted to string format.
        settings:

    Returns:
        np.array: A 2D array of with each row consisting of (max_bit_pos + prec) values

    Examples (assuming max_bit_pos=4 and settings.prec=3):
        0.123 --> 0000123
        1.23  --> 0001230
        1.2345 -> 0001234
    """
    base = float(settings.base)
    max_bit_pos = int(np.ceil(np.log(settings.max_val) / np.log(base)))

    before_decimals = []
    for i in range(max_bit_pos):
        digit = (val / base**(max_bit_pos - i - 1)).astype(int)
        before_decimals.append(digit)
        val -= digit * base**(max_bit_pos - i - 1)

    before_decimals = np.stack(before_decimals, axis=-1)

    if settings.prec > 0:
        after_decimals = []
        for i in range(settings.prec):
            digit = (val / base**(-i - 1)).astype(int)
            after_decimals.append(digit)
            val -= digit * base**(-i - 1)

        after_decimals = np.stack(after_decimals, axis=-1)
        digits = np.concatenate([before_decimals, after_decimals], axis=-1)

    return digits

# Test with example
settings = SerializerSettings()
num2bits(np.array([0.123, 1.23, 1.2345], dtype=np.float32), settings)

def serialize_series(val: np.array, settings: SerializerSettings, max_bit_pos: Optional[int] = None):
    """
    Serializes the series into strings to be used as an input to LLMs.

    Args:
        val: 1D-array Time series to be serialized.
        settings: Settings to use for serialization.
        max_bit_pos: Maximum number of bits before the decimal in bit representation of numbers.
    """
    assert val.ndim == 1, f'Expected 1D array, but got {val.ndim}D-array.'

    signs = 1 * (val >= 0) - 1 * (val < 0)
    is_missing = np.isnan(val)

    y_train_bits = num2bits(np.abs(val), settings)

    bit_strs = []
    for sign, missing, digit in zip(signs, is_missing, y_train_bits):

        # remove zeros
        nonzero_indices = np.where(digit != 0)[0]
        if len(nonzero_indices) == 0:
            digit = np.array([0])
        else:
            digit = digit[nonzero_indices[0]: ]

        # make a string for this digit by adding bit_sep and concatenating
        digit = ''.join([settings.bit_sep + str(b) for b in digit])

        # append the sign (+ or -)
        sign_sep = settings.plus_sign if sign == 1 else settings.minus_sign

        # missing
        if missing:
            bit_strs.append(settings.missing_str)
        else:
            bit_strs.append(sign_sep + digit)

    series = settings.time_sep.join(bit_strs) + settings.time_sep

    return series

time_series = y_train
scaler = get_scaler(time_series, 'basic')
vals = scaler.transform(time_series)
series_string = serialize_series(vals, settings)

print("Series:", vals[:10])
print("String:", series_string[:100])

"""### Step 3: Truncation/Pre-processing the Input

In this section of the tutorial, we focus on ensuring that the input number of tokens and the ***expected number of generated tokens*** do not exceed the model's context length. Given that time series representations can be extensive, the authors employ a strategy to manage this effectively:
1. **Tokenizer Access**
   - Access to a tokenizer is crucial to understand how our prompt will be tokenized.
   - Note: GPT-3.5/4 does not provide direct access to the tokenizer. We can use existing tokenizers as a fallback for approximation.
   - We will use `tiktoken` library for this purpose


2. **Iterative Selection for Input Series**
   - We will iteratively select the number of entries in the input time series.
   - This ensures that the output has the required number of entries and fits within the context length.

This approach allows us to tailor our time series data to fit within the constraints of the LLM, ensuring both the input and its generated output remain within the permissible token limit.

##### Tokenizer access
"""

def tokenize(string: str, model_name: str):
    """
    Tokenizes a given `string` as per the model specified by `model_name`.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    return encoding.encode(string)

series_tokens = tokenize(series_string, MODEL_NAME)
print(f"Total number of tokes in the series_string: {len(series_tokens)}. Context length: {CONTEXT_LENGTH}")
print(f"First 10 tokens: {series_tokens[:10]}")

"""Since the `series_string` is represented in 9K tokens, clearly we need to truncate it to fit it well within the context limit."""

# truncate input
def truncate_input(series_string: str, time_series: np.array, settings: SerializerSettings,
                   context_length: int, n_steps_to_predict: int = 12,
                   step_multiplier: float = 1.2, model_name: str = 'text-davinci-003', verbose: bool = True):
    """
    Truncates the string such that the total number of tokens in the input and the expected output does not exceed context length.

    Args:
        series_string: String for the time series as serialized by `serialize_series`.
        time_series: Original time series.
        settings: Settings for serialization of time series.
        context_length: Maximum number of tokens allowed in LLM.
        n_steps_to_predict: number of time steps in the future to predict.
        step_multiplier: A multiplier for estimation of tokens per time step in the output
        model_name: LLM to use.
        verbose:

    Returns:
        series_str (str): A string format of the truncated time series
        series (np.array): A truncated time series that will be used as an input
        avg_token_per_chunk (float): Number of tokens per time step
    """
    series_chunks = series_string.split(settings.time_sep)

    for i in range(len(series_chunks)):
        truncated_series_str = settings.time_sep.join(series_chunks[i:])
        if not truncated_series_str.endswith(settings.time_sep):
            truncated_series_str += settings.time_sep

        input_tokens = tokenize(truncated_series_str, model_name)

        num_input_tokens = len(input_tokens)
        avg_token_per_chunk = num_input_tokens / (len(series_chunks) - i)
        total_expected_output_tokens = n_steps_to_predict * avg_token_per_chunk * step_multiplier

        total_token_length = num_input_tokens + total_expected_output_tokens
        if total_token_length < context_length:
            truncated_time_series = time_series[i:]
            break

    if verbose:
        print(f"Discarding old {i} values in the series")
        print(f"Number of entries in the final input time series: {len(truncated_time_series)} representable in {num_input_tokens} tokens")
        print(f"Average tokens per time step: {avg_token_per_chunk}")

    return truncated_series_str, truncated_time_series

input_string, input_time_series = truncate_input(series_string, time_series, settings, CONTEXT_LENGTH, 20, 1.2, MODEL_NAME)

"""#### Preprocessing as a function

Let's package all of the above in a single function.
"""

def preprocess_series(series: np.array, scaler: Scaler, settings: SerializerSettings, truncate: bool,
                      context_length: int = 4097, n_steps_to_predict: int = 10, step_multiplier: float = 1.2,
                      model_name: str = 'text-davinci-003', verbose: bool = True):
    """
    Preprocesses the time series into the input for LLM.

    Args:
        series: Numerical time series to preprocess into the LLM prompt.
        scaler: Normalization function for the time series.
        settings: Serializer settings for the normalized time series.
        truncate: Whether to truncate the resulting string to fit the context length.
        context_length: Maximum number of tokens allowed in an LLM. Used only when truncate=True
        n_steps_to_predict: Number of future steps to predict. Used only when truncate=True.
        step_multiplier: A factor to estimate expected output tokens. Used only when truncate=True.
        model_name: LLM name.
        verbose:

    Returns:
        str: Preprocessed string for the truncated time series
        np.array: truncated time series
    """

    # Normalize the series
    normalized_series = scaler.transform(series)

    # Serialize the series
    series_str = serialize_series(normalized_series, settings)

    if not series_str.endswith(settings.time_sep):
        series_str += settings.time_sep

    # Truncate the series
    if truncate:
        truncated_series_str, truncated_series = truncate_input(series_str, series, settings, context_length, n_steps_to_predict, step_multiplier, model_name, verbose)
        return truncated_series_str, truncated_series

    return series_str, []

"""## Prompting LLM for Time Series Forecasting

We are now ready with the optimized prompt for time series forecasting. We can prompt the LLM to do time series forecasting.

For this purpose, we'll utilize the `completion/create` endpoint of OpenAI. For a comprehensive understanding of the various options available with this endpoint, please refer to the [OpenAI documentation](https://platform.openai.com/docs/api-reference/completions/create).

"""

def complete_series(model_name: str, series_string: str, settings: SerializerSettings, n_steps_to_predict: int = 12, temp: float = 0.7, n_samples: int = 5):
    """
    Calls LLMs to complete the series specified by series_string.

    Args:
       model_name: Type of LLM.
       series_string: Time series as represented by a string.
       settings: Serializer settings.
       n_steps_to_predict: number of time steps to predict
       temp: Temperature parameter for sampling the output. Refer to the API.
       n_samples: Number of output samples. Refer to the API.

    Returns:
        A list of `n_samples` number of strings sampled from the LLM
    """

    avg_token_per_step = len(tokenize(series_string, model_name)) / len(series_string.split(settings.time_sep))
    to_avoid_falling_short_factor = 1.05

    # Wherever possible, suppress unwanted outputs by exclusively selecting the relevant tokens
    # This is only possible in non-chat models such as GPT-3/GPT-2
    logit_bias = {}
    if model_name == 'text-davinci-003':
        allowed_tokens = [settings.bit_sep + str(i) for i in range(settings.base)]
        allowed_tokens += [settings.time_sep, settings.plus_sign, settings.minus_sign]
        allowed_tokens = [t for t in allowed_tokens if len(t) > 0]
        logit_bias = {tokenize(t, model_name)[0]: 30 for t in allowed_tokens}

    response = client.completions.create(
        model = model_name,
        prompt = series_string,
        max_tokens = int(avg_token_per_step * n_steps_to_predict * to_avoid_falling_short_factor),
        temperature = temp,
        logit_bias = logit_bias,
        n = n_samples,
        seed = 1234, # Used for reproducibility, but the results might still differ from one call to the other
    )

    return [choice.text for choice in response.choices]

scaler = get_scaler(y_train, 'basic', alpha=0.95)
settings = SerializerSettings()
n_steps_to_predict = 12
step_multiplier = 1.2

input_series_str, input_series = preprocess_series(y_train, scaler, settings, True, CONTEXT_LENGTH, n_steps_to_predict, step_multiplier, MODEL_NAME)

# input_string, input_time_series, avg_token_per_chunk = truncate_input(series_string, time_series, settings, CONTEXT_LENGTH, n_steps_to_predict, step_multiplier, MODEL_NAME)
sample_completions = complete_series(MODEL_NAME, input_series_str, settings, n_steps_to_predict, temp=0.7, n_samples=1)
print(sample_completions[0][:200])

"""## Post-processing LLM Outputs to Retrieve Numerical Time Series Data

After receiving outputs from the LLM, it's crucial to convert these back into a usable numerical format. This involves several steps:

1. **Deserializing/Preprocess the Output String**
   - Transform the output string back into a series format.
   - This involves parsing the string to isolate each data point.
   - Convert each element in the series from string to its numerical representation.


3. **Reversing the Scaling Process**
   - Use the previously applied scaler to convert the data back to its original scale.
   - This reverses the normalization or scaling process applied during the initial data preparation.

Thus, we can effectively transform the LLM's string outputs back into meaningful numerical time series data.

#### Deserializing/Preprocess the Output String

Read through the following function that converts the LLM output to the numerical series.
"""

# convert these predctions back to normal strings
def llm_output_to_series(bit_str: str, settings: SerializerSettings):
    """
    Converts the LLM output string to the numerical series.

    Args:
        bit_str: LLM's output
        settings: Serializer settings.
        n_steps_to_predict: number of time steps to predict.

    Returns:
        list of numerical values.
    """
    output_strs = bit_str.split(settings.time_sep)

    output_strs = [a for a in output_strs if len(a) > 0] # remove the empty ones
    output_strs = output_strs[:-1] # ignore the last one just so that the LLM stopped generating before the last one could be completed.

    signs, output_series = [], []
    for output in output_strs:

        # extracting string bits per time step
        if output.startswith(settings.minus_sign):
            sign = -1
            digit_str = output[len(settings.minus_sign):]
        else:
            sign = 1
            digit_str = output[len(settings.plus_sign):]

        # extract bits
        if settings.bit_sep == '':
            bits = [b for b in digit_str.strip()]
        else:
            bits = [b for b in digit_str.strip().split(settings.bit_sep)]

        # convert string bits to digits
        digits = [int(b) for b in bits]

        # convert the bits into numerical value in that base
        base = float(settings.base)
        D = len(digits)
        digits_flipped = np.flip(np.array(digits), axis=-1)
        powers = -np.arange(-settings.prec, -settings.prec + D)
        val = np.sum(digits_flipped/base**powers, axis=-1)

        if settings.half_bin_correction:
            val += 0.5/base**settings.prec

        output_series.append(sign * val)

    return np.array(output_series)

output_series = [llm_output_to_series(bit_str, settings) for bit_str in sample_completions]
output_series

"""### Reverting LLM Numerical Outputs to Original Scale

Once we have received the numerical output from the LLM, the next step is to transform this data back into the scale of the original time series.
For this, we will use the `scaler.inv_transform` method to invert the scaling transformation applied during preprocessing.

**Note on Sample Variability**: LLM outputs can vary in length for the final steps of the time series. To address this:
   - We compare the lengths of the series from various LLM samples.
   - We will retain only the shortest length across all samples to maintain a consistent series length.
"""

def rescale_series(predicted_series: List, scaler: Scaler):
    """
    Rescales the series back to the original scale.

    Args:
        predicted_series:
        scaler:

    Returns:
        np.array: Final predictions
    """
    predictions = []
    for a in output_series:
        predictions.append(scaler.inv_transform(a))

    return predictions

predictions = rescale_series(output_series, scaler)
predictions

"""### Metrics: How well did we do?

To asses how good are our predictions, we will use Mean Absolute Error (MAE) metric against the ground truth.
"""

def compute_mae(predictions, truth):
    """
    Computes MAE.
    """
    n_steps = min([len(l) for l in predictions])
    predictions = np.array([a[:n_steps] for a in predictions])
    abs_errors = np.abs(predictions - truth[:n_steps])
    return np.average(abs_errors)

compute_mae(predictions, y_test)

"""## Hyperparameter Tuning: How to select right parameters to prompt LLMs?

We have the working pipeline to perform forecasting. However, we still need to decide some of the parameters that will boost our time series forecasting performance.
As a result, when working with LLMs for time series forecasting, fine-tuning hyperparameters is a crucial step. Here's a walkthrough of how to approach this:

1. **Choosing Hyperparameters**:
   - We have the following hyperparameters:
     - $\alpha$: Scaling parameter for pre-processing time series.
     - $\beta$: Offset parameter for pre-processing time series.
     - `prec`: String representation parameter. Higher precision restricts the input length of time series.
     - `temp`: LLM sampling parameter. Alternatively, use `top_p`. Refer to the documentation for more details.


2. **Setting a Benchmark**:
   - We'll use the Negative Log-Likelihood per Dimension (NLL/D) as our evaluation metric, based on the likelihood of the validation series given the training data, as detailed in Appendix A.2 of the Gruver et al. 2023.


3. **Process for Tuning**:
   - Divide the training data into two parts: training and validation.
   - Test different combinations of hyperparameters on this split data. Our goal is to identify the combination that minimizes the NLL/D.

### Negative Log-likelihood per Dimension (NLL/D)

Detailed steps for this calculation will be provided, including a function in the subsequent section for practical implementation.

#### Train-Val data split
"""

scaler = get_scaler(y_train, 'advanced', alpha=0.95, beta=0.3)
settings = SerializerSettings()
step_multiplier = 1.2
val_length = 30 # validation step requires predicting `val_length` steps into the future

train_series = y_train[:-val_length]
validation_series = y_train[-val_length:]

train_series_str, train_series_arr = preprocess_series(train_series, scaler, settings, True, CONTEXT_LENGTH, val_length, step_multiplier)


validation_series_str, _ =  preprocess_series(validation_series, scaler, settings, False)

"""#### Querying LLM to predict the likelihood of validation string conditioned on train string

To make this query, we will need to specify the following
1. `max_tokens = 0`: Don't generate new tokens
2. `logprobs=5`: Send the log probability of top 5 tokens per predicted token
3. `echo=True`: Returns the input prompt in the output.
3. `temperature=1`: This is a hyperparameter to be selected.   
"""

# Query the LLM to extract the logits for each token
full_series = train_series_str + validation_series_str

response = client.completions.create(
    model = MODEL_NAME,
    prompt = full_series,
    logprobs = 5,
    max_tokens = 0,
    echo = True,
    temperature = 1.0,
)

"""### Computing Negative Log-Likelihood per Dimension (NLL/D)

The following two code blocks are there to help you understand the computation thorugh DIY. The entire computation is encapsulated in the function. One can refer to that function to check the right answers.

##### Key Attributes in `response.choices[0].logprobs`
1. **`token_logprobs`**: Contains log probabilities of predicted tokens.
2. **`tokens`**: Tokens at each prediction step.
3. **`echo=True`**:
   - Requests the API to return the input prompt as tokenized by the LLM.
   - As a result, log probability for each token is also returned, crucial for assessing the likelihood of observing the validation series.

4. **`top_logprobs`**:
   - Shows top log probabilities for tokens at each step.
   - Useful for understanding possible variations in LLM outputs.

Delve into these attributes for a deeper understanding of the LLM's functioning.

##### Note on Time Steps
- Since LLM outputs don’t explicitly mark time steps, you'll need to infer the start of the validation string. This is done by matching the cumulative count of `settings.time_sep` with the length of the input series.

##### Computing Procedure
1. **Extract Log Probabilities**: Focus on tokens that pertain to the validation series.
2. **Probability Adjustment**: Account for extraneous tokens. This involves tweaking probabilities to reflect modifications (like logit_bias adjustments) made to the model.
3. **Aggregate Log Probabilities**: Combine these to represent the overall probability of predicting the validation series.
4. **Continuous Distribution Adjustment**: Uniformly distribute likelihood over the bins in predicted ranges.
5. **Final Adjustment**: Convert this probability back to the scale of the original input.

This approach ensures an efficient and accurate calculation of NLL/D, providing valuable insights into your model's performance.
"""

logprobs = np.array(response.choices[0].logprobs.token_logprobs, dtype=np.float32)
output_string_arr = np.array(response.choices[0].logprobs.tokens)
top5logprobs = response.choices[0].logprobs.top_logprobs

# Compute the starting point for the output string
seps = output_string_arr == settings.time_sep
val_start = np.argmax(np.cumsum(seps) == len(train_series_arr)) + 1


# We are only interested in computing likelihood for the tokens predicted for validation_series
val_logprobs = logprobs[val_start:]
val_top5logprobs = top5logprobs[val_start:]

"""**Note:** Pay attention to how the log probability of the chosen token might differ from the one listed in `top_logprobs`. The `logprobs` represent the log probability of tokens in `output_string_arr`, which corresponds to the validation series itself."""

"".join(output_string_arr[val_start:]), validation_series_str, val_top5logprobs[2], val_logprobs[2]

# Probability Adjustment: adjust logprobs by removing extraneous tokens
# Note: This adjustment rewards the model to think solely in terms of alllowed tokens
allowed_tokens = [settings.bit_sep + str(i) for i in range(settings.base)]
allowed_tokens += [settings.time_sep, settings.plus_sign, settings.minus_sign, settings.bit_sep]
allowed_tokens = {t for t in allowed_tokens if len(t) > 0}

p_extra = []
for i in range(len(val_top5logprobs)):
    x = sum(np.exp(ll) for k,ll in val_top5logprobs[i].items() if not (k in allowed_tokens))
    p_extra.append(x)
p_extra = np.array(p_extra)

if settings.bit_sep == '':
    p_extra = 0

adjusted_val_logprobs = val_logprobs - np.log(1-p_extra)

# Aggregate Log Probabilities: Compute total logprobs per dimension
digit_bits = -adjusted_val_logprobs.sum()
loglikelihood_per_dimension = digit_bits/len(validation_series)

# Continuous Distribution Adjustment: Adjust the discrete likelihood to continuous distribution by assuming bin over the range (See Page 5, 1st Para)
transformed_nll = loglikelihood_per_dimension - settings.prec * np.log(settings.base)

# Final Adjustment: Adjust the likelihood for input scaling (See Page 5, 1st Para)
avg_logdet_dydx = np.log(vmap(grad(scaler.transform))(validation_series)).mean()

nlld = transformed_nll - avg_logdet_dydx
print(nlld)

def compute_nlld(model_name: str, train_series_str: str, len_train_series: int, validation_series_str: str, validation_series: int,
                 settings: SerializerSettings, scaler: Scaler, val_length: int, temp: float):
    """
    Computes NLL/D metric based on how likely is the target_series conditioned on the input_series.

    Args:
        model_name: LLM to use
        train_series_str: Serialized input_series
        len_train_series: Number of time steps in the train time series.
        validation_series: validation series to be used for transformation.
        len_validation_series: Number of time steps in the validation time series
        settings: Serialization settings.
        scaler: Scaler to define transformation of data.
        val_length: Total number of steps to predict and check against the validation series.
        temp: Temperature parameter for sampling in LLM.

    Returns:
        float: NLL/D value.
    """
    # Use LLM for zero-shot predictions
    full_series = train_series_str + validation_series_str
    response = client.completions.create(
        model = model_name,
        prompt = full_series,
        logprobs = 5,
        max_tokens = 0,
        echo = True, # Send back the input string too.
        temperature = temp,
        seed = 1234, # for reproducibility
    )

    logprobs = np.array(response.choices[0].logprobs.token_logprobs, dtype=np.float32)
    output_string_arr = np.array(response.choices[0].logprobs.tokens) # expect the full string since echo=True
    top5logprobs = response.choices[0].logprobs.top_logprobs

    # Compute the starting point for the output string
    seps = output_string_arr == settings.time_sep
    val_start = np.argmax(np.cumsum(seps) == len_train_series) + 1

    val_logprobs = logprobs[val_start:]
    val_top5logprobs = top5logprobs[val_start:]

    # adjust logprobs by removing extraneous tokens
    ## Note: This adjustment rewards the model to think solely in terms of alllowed tokens
    allowed_tokens = [settings.bit_sep + str(i) for i in range(settings.base)]
    allowed_tokens += [settings.time_sep, settings.plus_sign, settings.minus_sign, settings.bit_sep]
    allowed_tokens = {t for t in allowed_tokens if len(t) > 0}

    p_extra = np.array([sum(np.exp(ll) for k,ll in val_top5logprobs[i].items() if not (k in allowed_tokens)) for i in range(len(val_top5logprobs))])

    if settings.bit_sep == '':
        p_extra = 0

    adjusted_val_logprobs = val_logprobs - np.log(1-p_extra)

    # Compute total logprobs per dimension
    digit_bits = -adjusted_val_logprobs.sum()
    loglikelihood_per_dimension = digit_bits/len(validation_series)

    # Adjust the discrete likelihood to continuous distribution (See Page 5, 1st Para)
    transformed_nll = loglikelihood_per_dimension - settings.prec * np.log(settings.base)

    # Adjust the likelihood for input scaling (See Page 5, 1st Para)
    avg_logdet_dydx = np.log(vmap(grad(scaler.transform))(validation_series)).mean()

    return transformed_nll - avg_logdet_dydx

scaler = get_scaler(train_series, 'advanced', alpha=0.95, beta=0.3)
settings = SerializerSettings(prec=3)
temp = 1.0

val_nll = compute_nlld(MODEL_NAME, train_series_str, len(train_series_arr),
             validation_series_str, validation_series,
             settings, scaler, val_length, temp)

print(val_nll)

"""## Hyperparameter Tuning: Selecting the best parameters based on NLL/D

We will perform a grid search over the parameters.

**Caveat:** Depending on the number of hyperparameters, the API will be called that many times. This step might end up consuming a lot of API's credits.
"""

from sklearn import model_selection

params = {
    'alpha': [0.2, 0.5, 0.8],
    'beta': [0, 0.15],
    'prec': [2],
    'temp': [1.0]
}

hyperparams = model_selection.ParameterGrid(params)

best_val_nll = float('inf')
best_hypers = None
hyper_performance = []

for param in hyperparams:
    scaler = get_scaler(train_series, 'advanced', alpha=param['alpha'], beta=param['beta'])
    settings = SerializerSettings(prec=param['prec'])

    # serialize input_series
    train_series_str, train_series_arr = preprocess_series(train_series, scaler, settings, True, CONTEXT_LENGTH, val_length, step_multiplier, MODEL_NAME)
    validation_series_str, _ =  preprocess_series(validation_series, scaler, settings, False)

    val_nll = compute_nlld(MODEL_NAME, train_series_str, len(train_series_arr),
                 validation_series_str, validation_series,
                 settings, scaler, val_length, param['temp'])

    if val_nll < best_val_nll:
        best_val_nll = val_nll
        best_hypers = param
        best_hypers['scaler'] = scaler

    hyper_performance.append((param, val_nll))
    print(f'Hyper: {param}. Val NLL: {val_nll}\n')


print(f'Best Hyperparameters: {best_hypers} Vall NLL: {best_val_nll}')

perf = []
for k, v in hyper_performance:
    perf.append({
        'alpha': k['alpha'],
        'beta': k['beta'],
        'prec': k['prec'],
        'temp': k['temp'],
        'nll': v
    })
perf = pd.DataFrame(perf)

perf.sort_values(by='nll')

"""## Make predctions using the best hyperparameters

Finally, we will use the best hyperparameters for time series forecasting.


"""

input_series = y_train
n_steps_to_predict = 30
step_multiplier = 1.2
HYPERS = {
    'alpha': 0.2,
    'beta': 0.15,
    'prec': 2,
    'temp': 1.0
}

scaler = get_scaler(train_series, 'advanced', alpha=HYPERS['alpha'], beta=HYPERS['beta'])
settings = SerializerSettings(prec=HYPERS['prec'])

input_series_str, input_series_arr = preprocess_series(input_series, scaler, settings, True, CONTEXT_LENGTH, n_steps_to_predict, step_multiplier, MODEL_NAME)

sample_completions = complete_series(MODEL_NAME, input_series_str, settings, n_steps_to_predict, temp=HYPERS['temp'], n_samples=10)

output_series = [llm_output_to_series(bit_str, settings) for bit_str in sample_completions]
predictions = rescale_series(output_series, scaler)

minimum_steps_predicted = min([len(l) for l in predictions])
print("\nMinimum steps predicted across samples:", minimum_steps_predicted)
predictions = np.array([a[:minimum_steps_predicted] for a in predictions])

point_estimates = np.median(predictions, axis=0)
print("\nestimates:\n", point_estimates)

benchmarks = pd.read_csv('predictions.csv', index_col=0)[: minimum_steps_predicted]

benchmarks['GPT-3'] = point_estimates

print('\n')
for col in benchmarks.columns:
    if col == 'test':
        continue
    MAE = np.average(np.abs(benchmarks[col] - benchmarks['test']))
    print(f"MAE: {MAE: 0.5f} \t Model: {col}")

"""#### Reflection and Inquiry

1. **Variability of Seed Parameter**: Reflect on how the seed parameter impacts the call's output. What variations do you notice when the seed isn't set?

2. **Temperature Parameter's Influence**: How does the temperature setting affect the model's estimates? What changes occur when the temperature is set very low?

3. **Impact of `prec` on Forecasts**: Examine how the precision (`prec`), particularly the number of digits after the decimal point, influences the forecasts.

4. **Cost-Effectiveness of NLL/D vs. MAE**: Why might NLL/D be a more cost-effective metric for validation compared to MAE?

5. **Adaptations for Different Models**: Consider how you would adjust this pipeline for other models like GPT-3.5/4 or LLaMA.
"""
