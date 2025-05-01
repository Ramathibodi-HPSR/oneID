import pandas as pd
import numpy as np
from tabulate import tabulate
from pandas.api.types import is_numeric_dtype
from scipy import stats
from sklearn.utils import resample
from scipy.stats import shapiro, kruskal, chi2_contingency, anderson
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from tqdm import tqdm
from docx import Document
from docx.shared import Inches
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

def add_to_docx(results, table_name, output_dir):
    doc = Document()
    
    # Add a title for the table
    doc.add_heading(table_name, level=2)
    
    # Check if results is a DataFrame or list and if it's empty
    if isinstance(results, pd.DataFrame):
        if results.empty:
            doc.add_paragraph("No data available.")
        else:
            # Convert DataFrame to list of dictionaries
            results = results.to_dict(orient='records')
    elif not results:
        doc.add_paragraph("No data available.")
        return

    # Add a table with a header row if there is data
    if results:
        headers = results[0].keys()
        table = doc.add_table(rows=1, cols=len(headers))
        hdr_cells = table.rows[0].cells
        for i, header in enumerate(headers):
            hdr_cells[i].text = header

        # Add the data rows
        for row_data in results:
            row_cells = table.add_row().cells
            for i, header in enumerate(headers):
                row_cells[i].text = str(row_data[header])
    
    # Save the document
    doc.save(f'{output_dir}/{table_name}.docx')



def des_cat(df_input, factor, group_var=None, p_value=None):
    rows = []
    
    if group_var:
        df = df_input.dropna(subset=[group_var])

    # Header row
    header_row = {'Characteristics': factor, 'Total': '', 'p-value': p_value}
    if group_var:
        groups = df[group_var].dropna().unique()
        for group in groups:
            header_row[group] = ''
    rows.append(header_row)

    # Calculate total for each category
    categories = df[factor].dropna().unique()
    grand_total = len(df[factor].dropna())

    for category in categories:
        subtotal = len(df[df[factor] == category])
        percent = (subtotal / grand_total) * 100
        row = {'Characteristics': category, 'Total': f'{subtotal:,}\n({percent:.2f}%)'}

        if group_var:
            for group in groups:
                group_total = len(df[(df[group_var] == group) & (df[factor].notna())])
                group_count = len(df[(df[factor] == category) & (df[group_var] == group)])
                group_percent = (group_count / group_total) * 100
                row[group] = f'{group_count:,}\n({group_percent:.2f}%)'

        row['p-value'] = ''  # Add p-value column placeholder
        rows.append(row)

    return rows


def des_num(df, factor, group_var=None, p_value=None, is_normal=True):
    rows = {}

    if is_normal:
        mean = df[factor].dropna().mean()
        sd = df[factor].dropna().std()
        rows = {
            'Characteristics': factor,
            'Total': f'{mean:.2f}\n({sd:.2f})',
            'p-value': p_value
        }

        if group_var:
            groups = df[group_var].dropna().unique()
            for group in groups:
                group_df = df[df[group_var] == group]
                submean = group_df[factor].dropna().mean()
                subsd = group_df[factor].dropna().std()
                rows[group] = f'{submean:.2f}\n({subsd:.2f})'

    else:
        median = df[factor].dropna().median()
        p25 = df[factor].dropna().quantile(0.25)
        p75 = df[factor].dropna().quantile(0.75)
        rows = {
            'Characteristics': factor,
            'Total': f'{median:.2f}\n({p25:.2f}-{p75:.2f})',
            'p-value': p_value
        }

        if group_var:
            groups = df[group_var].dropna().unique()
            for group in groups:
                group_df = df[df[group_var] == group]
                submedian = group_df[factor].dropna().median()
                subp25 = group_df[factor].dropna().quantile(0.25)
                subp75 = group_df[factor].dropna().quantile(0.75)
                rows[group] = f'{submedian:.2f}\n({subp25:.2f}-{subp75:.2f})'

    return rows



def describe(df, factors, group_var, overall=False, bootstrap=False, n_bootstraps=1000, table_name=None):
    
    print(f"{table_name}")
    
    results = pd.DataFrame(columns=["Characteristics", "Total"] + list(df[group_var].dropna().unique()) + ["p-value"])

    for factor in factors:

        if df[factor].dtype == 'O':
            # Chi-square with Bootstrapping and Downsampling
            contingency_table = pd.crosstab(df[factor], df[group_var])
            group_sizes = contingency_table.sum(axis=0)
            min_group_size = group_sizes.min()
            
            if bootstrap:
                p_values = []
                for _ in tqdm(range(n_bootstraps), desc=f"Bootstrapping {factor}", leave=False):
                    # Downsample each group to the size of the smallest group
                    sampled_df = pd.concat([df[df[group_var] == group].sample(min_group_size, replace=True) 
                                            for group in df[group_var].dropna().unique()])
                    sampled_contingency = pd.crosstab(sampled_df[factor], sampled_df[group_var])
                    _, p_val, _, _ = chi2_contingency(sampled_contingency)
                    p_values.append(p_val)
                p_value_mean = np.mean(p_values)
                ci_lower, ci_upper = np.percentile(p_values, [2.5, 97.5])
                p_value = f"{p_value_mean:.2f}\n(95% CI: {ci_lower:.2f}-{ci_upper:.2f})"
            else:
                contingency_table = pd.crosstab(df[factor], df[group_var])
                _, p_value, _, _ = chi2_contingency(contingency_table)
                p_value = f"{p_value:.2f}"

            des = des_cat(df, factor, group_var, p_value)
            descriptive_df = pd.DataFrame(des)

        elif is_numeric_dtype(df[factor]):
            normality_pvalues = {}
            for group in df[group_var].dropna().unique():
                group_data = df[df[group_var] == group][factor].dropna()
                if len(group_data) > 5000:
                    ad_stat, critical_values, _ = anderson(group_data)
                    pvalue = (ad_stat > critical_values[-1])
                    normality_pvalues[group] = 0 if pvalue else 1
                else:
                    if len(group_data) > 3:
                        stat, pvalue = shapiro(group_data)
                        normality_pvalues[group] = pvalue
            
            is_normal = all(p > 0.05 for p in normality_pvalues.values())
            groups = [df[df[group_var] == group][factor].dropna() for group in df[group_var].dropna().unique()]
            min_group_size = min([len(group) for group in groups])

            if is_normal:
                if len(groups) == 2 and all(len(group) > 1 for group in groups):
                    if bootstrap:
                        p_values = []
                        for _ in tqdm(range(n_bootstraps), desc=f"Bootstrapping {factor}", leave=False):
                            sample_group1 = resample(groups[0], n_samples=min_group_size, replace=True)
                            sample_group2 = resample(groups[1], n_samples=min_group_size, replace=True)
                            _, p_val = stats.ttest_ind(sample_group1, sample_group2, equal_var=False)
                            p_values.append(p_val)
                        p_value_mean = np.mean(p_values)
                        ci_lower, ci_upper = np.percentile(p_values, [2.5, 97.5])
                        p_value = f"{p_value_mean:.2f}\n(95% CI: {ci_lower:.2f}-{ci_upper:.2f})"
                    else:
                        sample_group1 = resample(groups[0], n_samples=min_group_size, replace=False)
                        sample_group2 = resample(groups[1], n_samples=min_group_size, replace=False)
                        _, p_value = stats.ttest_ind(sample_group1, sample_group2, equal_var=False)
                        p_value = f"{p_value:.2f}"
                    print(f'Testing for {factor} using T-test')
                elif len(groups) > 2 and all(len(group) > 1 for group in groups):
                    if bootstrap:
                        p_values = []
                        for _ in tqdm(range(n_bootstraps), desc=f"Bootstrapping {factor}", leave=False):
                            sampled_groups = [resample(group, n_samples=min_group_size, replace=True) for group in groups]
                            _, p_val = stats.f_oneway(*sampled_groups)
                            p_values.append(p_val)
                        p_value_mean = np.mean(p_values)
                        ci_lower, ci_upper = np.percentile(p_values, [2.5, 97.5])
                        p_value = f"{p_value_mean:.2f}\n(95% CI: {ci_lower:.2f}-{ci_upper:.2f})"
                    else:
                        sampled_groups = [resample(group, n_samples=min_group_size, replace=False) for group in groups]
                        _, p_value = stats.f_oneway(*sampled_groups)
                        p_value = f"{p_value_mean:.2f}"
                    print(f'Testing for {factor} using ANOVA')
                else:
                    p_value = float('nan')
            else:
                if len(groups) == 2 and all(len(group) > 1 for group in groups):
                    if bootstrap:
                        p_values = []
                        for _ in tqdm(range(n_bootstraps), desc=f"Bootstrapping {factor}", leave=False):
                            sample_group1 = resample(groups[0], n_samples=min_group_size, replace=True)
                            sample_group2 = resample(groups[1], n_samples=min_group_size, replace=True)
                            _, p_val = stats.mannwhitneyu(sample_group1, sample_group2)
                            p_values.append(p_val)
                        p_value_mean = np.mean(p_values)
                        ci_lower, ci_upper = np.percentile(p_values, [2.5, 97.5])
                        p_value = f"{p_value_mean:.2f}\n(95% CI: {ci_lower:.2f}-{ci_upper:.2f})"
                    else:
                        sample_group1 = resample(groups[0], n_samples=min_group_size, replace=False)
                        sample_group2 = resample(groups[1], n_samples=min_group_size, replace=False)
                        _, p_value = stats.mannwhitneyu(sample_group1, sample_group2)
                        p_value = f"{p_value:.2f}"
                    print(f'Testing for {factor} using Mann-Whitney U')
                elif len(groups) > 2 and all(len(group) > 1 for group in groups):
                    if bootstrap:
                        p_values = []
                        for _ in tqdm(range(n_bootstraps), desc=f"Bootstrapping {factor}", leave=False):
                            sampled_groups = [resample(group, n_samples=min_group_size, replace=True) for group in groups]
                            _, p_val = stats.kruskal(*sampled_groups)
                            p_values.append(p_val)
                        p_value_mean = np.mean(p_values)
                        ci_lower, ci_upper = np.percentile(p_values, [2.5, 97.5])
                        p_value = f"{p_value_mean:.2f}\n(95% CI: {ci_lower:.2f}-{ci_upper:.2f})"
                    else:
                        _, p_value = stats.kruskal(*groups)
                        p_value = f"{p_value:.2f}"
                    print(f'Testing for {factor} using Kruskal-Walliss H')
                else:
                    p_value = f"N/A"

            des = des_num(df, factor, group_var, p_value, is_normal)
            descriptive_df = pd.DataFrame([des])

        results = pd.concat([results, descriptive_df], ignore_index=True)

    if not overall:
        results.drop(columns=['Total'], inplace=True)

    print(tabulate(results, showindex=False, headers="keys"))
    add_to_docx(results, table_name, output_dir='output/analyse')


def cronbach_alpha(df):
    
    # Number of items
    n_items = df.shape[1]
    
    # Variance of each item
    item_variances = df.var(axis=0, ddof=1)
    
    # Variance of the total score
    total_score_variance = df.sum(axis=1).var(ddof=1)
    
    # Cronbach's alpha formula
    alpha = (n_items / (n_items - 1)) * (1 - (item_variances.sum() / total_score_variance))
    
    return alpha


def logistic_regression(df_input, outcome_column, outcome_choice, predictors, reference_group=None, multivariate=False):
    
    df = df_input.copy()
    df = df[predictors + [outcome_column]]

    table_name = f"logis_{outcome_choice}_vs_others"
    print(f"Logistic regression analysis for {outcome_choice} as outcome")

    # Create a binary outcome based on the user's choice
    df['outcome'] = df[outcome_column].apply(lambda x: 1 if x == outcome_choice else 0)

    # Ensure predictors are numeric and handle categorical data
    categorical_predictors = df[predictors].select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Handling reference group before creating dummies
    if reference_group:
        for predictor, ref_value in reference_group.items():
            if predictor in categorical_predictors:
                df[predictor] = pd.Categorical(df[predictor], categories=[ref_value] + [x for x in df[predictor].unique() if x != ref_value], ordered=True)     

    df = pd.get_dummies(df, columns=categorical_predictors, drop_first=True)
    df = df.applymap(lambda x: int(x) if isinstance(x, bool) else x)

    # Determine final list of predictors including dummy variables
    predictors_dummies = []
    for predictor in predictors:
        if predictor in categorical_predictors:
            
            ref_value = reference_group.get(predictor)
            # Add a reference row for the original category
            ref_row = {
                "Variables": f"{predictor}_{ref_value}",
                "u-var OR": "reference",
                "u-var 95% CI" : "",
                "p-value": "",
                "m-var OR": "reference",
                "m-var 95% CI": "",
                "m-var p-value": ""
            }
            predictors_dummies.append(ref_row)
            # Add all dummy columns corresponding to this categorical predictor
            dummies = df.columns[df.columns.str.startswith(predictor)]
            predictors_dummies.extend(dummies)
        else:
            predictors_dummies.append(predictor)
    
    results = []

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Univariate analysis
    for predictor in predictors_dummies:
        if isinstance(predictor, str):  # Exclude the reference row
            X = df[[predictor]]
            X = sm.add_constant(X)
            y = df['outcome']
            try:
                model = sm.Logit(y, X)
                result = model.fit(disp=0)  # Suppress output
                
                coef = np.exp(result.params[predictor])
                conf = np.exp(result.conf_int().loc[predictor])
                pvalue = result.pvalues[predictor]
                ci = f"{conf[0]:.2f}, {conf[1]:.2f}"
                row = {
                    "Variables": predictor,
                    "u-var OR": f"{coef:.2f}",
                    "u-var 95% CI" : ci,
                    "p-value": f"{pvalue:.3f}",
                    "m-var OR": "",
                    "m-var 95% CI": "",
                    "m-var p-value": ""
                }
                results.append(row)
            except Exception as e:
                print(f"Error fitting univariate model for {predictor}: {e}")
                continue
        else:
            # Append reference row as is
            results.append(predictor)

    # Multivariate analysis
    if multivariate:
        X = df[[pred for pred in predictors_dummies if isinstance(pred, str)]]
        X = sm.add_constant(X)
        y = df['outcome']
        try:
            model = sm.Logit(y, X)
            result = model.fit(disp=0)  # Suppress output

            for predictor in predictors_dummies:
                if isinstance(predictor, str):  # Exclude the reference row
                    coef = np.exp(result.params[predictor])
                    conf = np.exp(result.conf_int().loc[predictor])
                    pvalue = result.pvalues[predictor]
                    ci = f"{conf[0]:.2f}, {conf[1]:.2f}"
                    
                    # Find the row in the results and update it
                    for row in results:
                        if row["Variables"] == predictor:
                            row["m-var OR"] = f"{coef:.2f}"
                            row['m-var 95% CI'] = ci
                            row["m-var p-value"] = f"{pvalue:.3f}"
                            break
        except Exception as e:
            print(f"Error fitting multivariate model: {e}")

    # Display the results
    print(tabulate(results, headers="keys"))
    add_to_docx(results, table_name, output_dir='output/analyse')

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

def interrupted_time_series(df_input, time_column, outcome_column=None, intervention_point=None, 
                            control_columns=None, time_unit='date', show_summary=True, plot=True, ax=None, point_size=None, 
                            title='Title', axis_label_size=14, title_size=16, tick_label_size=12, line_width=2, y_lim=None, first_col=False,
                            counterfactual_line=True):
    """
    Perform Interrupted Time Series analysis.
    
    Parameters:
    - df: pd.DataFrame, the data containing the time series.
    - time_column: str, the column name representing time.
    - outcome_column: str, optional, the column name representing the outcome variable. If None, use counts.
    - intervention_point: int/str/datetime, optional, the time point at which the intervention occurred.
    - control_columns: list of str, optional, names of control variables.
    - time_unit: str, optional, the unit of time ('minute', 'hour', 'day', 'month', etc.). Default is 'day'.
    - plot: bool, optional, whether to plot the time series and regression lines.
    
    Returns:
    - results: Regression results summary from statsmodels.
    """

    df = df_input.copy()

    # Convert the time column to a datetime format
    df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
    
    # Check for any non-datetime values and handle them (e.g., drop or fill with a default value)
    if df[time_column].isnull().any():
        print(f"Warning: {df[time_column].isnull().sum()} non-datetime entries found in {time_column}. These will be dropped.")
        df = df.dropna(subset=[time_column])
    
    # Manipulate the time column based on the specified time unit
    if time_unit == 'date':
        df[time_column] = df[time_column].dt.floor('D')  # Only keep the date part
    elif time_unit == 'month':
        df[time_column] = df[time_column].dt.to_period('M').dt.to_timestamp()  # Keep only month and year
    elif time_unit == 'year':
        df[time_column] = df[time_column].dt.to_period('Y').dt.to_timestamp()  # Keep only the year
    # For 'minute', 'hour', 'week', and other time units, no further manipulation is needed
    
    if outcome_column is None:
        df = df.groupby(time_column).size().reset_index(name='outcome')
        outcome_column = 'outcome'
    else:
        df = df.groupby(time_column).agg({outcome_column: 'mean'}).reset_index()
    
    # Create time variable based on the specified time unit
    time_conversion = {
        'minute': 'T',
        'hour': 'H',
        'date': 'D',
        'week': 'W',
        'month': 'M',
        'year': 'Y'
    }
    df['time'] = df[time_column].dt.to_period(time_conversion.get(time_unit, 'D')).astype(str)
    df['time'] = pd.to_datetime(df['time']).rank(method='first').astype(int)

    # Set the intervention point
    if intervention_point is not None:
        if isinstance(intervention_point, str) or isinstance(intervention_point, pd.Timestamp):
            intervention_point = df[df[time_column] >= pd.to_datetime(intervention_point)].iloc[0]['time']

    # Create pre- and post-intervention indicators
    df['intervention'] = (df['time'] >= intervention_point).astype(int) if intervention_point else 0
    
    # Create time after intervention variable
    df['time_after_intervention'] = df['time'] - df['time'][df['intervention'] == 1].min()
    df['time_after_intervention'] = df['time_after_intervention'].apply(lambda x: x if x >= 0 else 0)
    
    # Create design matrix
    X = sm.add_constant(df[['time', 'intervention', 'time_after_intervention']])
    
    # Include control variables if provided
    if control_columns:
        X = sm.add_constant(df[['time', 'intervention', 'time_after_intervention'] + control_columns])
    
    # Fit the model
    model = sm.OLS(df[outcome_column], X)
    results = model.fit()
    
    if show_summary:
        print(results.summary())
    
    if plot:
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the time series with intervention
        ax.scatter(df['time'], df[outcome_column], label='Outcome', color='grey', alpha=0.5, s=point_size)
        
        if intervention_point:
            ax.axvline(x=df['time'][df['time'] == intervention_point].iloc[0], color='red', linestyle='--', label='Intervention Point', linewidth=line_width)
        
        if counterfactual_line and intervention_point:
            # Create the counterfactual line by extending the pre-intervention trend
            pre_intervention_model = sm.OLS(df[outcome_column][df['time'] < intervention_point], 
                                            sm.add_constant(df[['time']][df['time'] < intervention_point])).fit()
            df['counterfactual'] = pre_intervention_model.predict(sm.add_constant(df[['time']]))
            ax.plot(df['time'], df['counterfactual'], label='Counterfactual', color='green', linestyle='--', linewidth=line_width)
        
        df['predicted'] = results.predict(X)
        ax.plot(df['time'], df['predicted'], label='Fitted values', color='blue', linewidth=line_width)
        if y_lim is not None:  # Only set y_lim if it is provided
            ax.set_ylim(0, y_lim)
        ax.set_xlabel(time_unit, fontsize=axis_label_size)
        if first_col:
            ax.set_ylabel(outcome_column, fontsize=axis_label_size)
        ax.set_title(title, fontsize=title_size)
        ax.tick_params(axis='both', labelsize=tick_label_size)
        ax.grid(True)
    
    return results


def interrupted_time_series_with_counterfactual(df_input, time_column, outcome_column=None, intervention_point=None, 
                                                                  split_point=None, control_columns=None, time_unit='date', 
                                                                  show_summary=True, plot=True, ax=None, point_size=10, 
                                                                  title='Title', axis_label_size=14, title_size=16, 
                                                                  tick_label_size=12, line_width=2, y_lim=(0,1), first_col=False,
                                                                  counterfactual_postanalysis=True):
    """
    Perform Interrupted Time Series analysis with a split for counterfactual comparison and additional post-intervention regression.
    
    Parameters:
    - df_input: pd.DataFrame, the data containing the time series.
    - time_column: str, the column name representing time.
    - outcome_column: str, optional, the column name representing the outcome variable. If None, use counts.
    - intervention_point: int/str/datetime, optional, the time point at which the intervention occurred.
    - split_point: int/str/datetime, optional, the time point at which to split the data for counterfactual comparison.
    - control_columns: list of str, optional, names of control variables.
    - time_unit: str, optional, the unit of time ('minute', 'hour', 'day', 'month', etc.). Default is 'day'.
    - plot: bool, optional, whether to plot the time series and regression lines.
    - counterfactual_line: bool, optional, whether to plot the counterfactual (pre-intervention trend) in the post-intervention period.

    Returns:
    - results_pre_split: Regression results summary from pre-split analysis.
    - results_post_split: Regression results summary from post-split analysis.
    """

    df = df_input.copy()

    # Convert the time column to a datetime format
    df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
    
    # Check for any non-datetime values and handle them (e.g., drop or fill with a default value)
    if df[time_column].isnull().any():
        print(f"Warning: {df[time_column].isnull().sum()} non-datetime entries found in {time_column}. These will be dropped.")
        df = df.dropna(subset=[time_column])
    
    # Manipulate the time column based on the specified time unit
    if time_unit == 'date':
        df[time_column] = df[time_column].dt.floor('D')  # Only keep the date part
    elif time_unit == 'month':
        df[time_column] = df[time_column].dt.to_period('M').dt.to_timestamp()  # Keep only month and year
    elif time_unit == 'year':
        df[time_column] = df[time_column].dt.to_period('Y').dt.to_timestamp()  # Keep only the year
    # For 'minute', 'hour', 'week', and other time units, no further manipulation is needed
    
    if outcome_column is None:
        df = df.groupby(time_column).size().reset_index(name='outcome')
        outcome_column = 'outcome'
    else:
        df = df.groupby(time_column).agg({outcome_column: 'mean'}).reset_index()
    
    # Create time variable based on the specified time unit
    time_conversion = {
        'minute': 'T',
        'hour': 'H',
        'date': 'D',
        'week': 'W',
        'month': 'M',
        'year': 'Y'
    }
    # Get the maximum and minimum time
    max_time = df[time_column].max()
    min_time = df[time_column].min()

    # Create the time range DataFrame
    its_time_range = pd.date_range(start=min_time, end=max_time, freq=time_conversion.get(time_unit, 'D'))
    its_df = pd.DataFrame({time_column: its_time_range})
    its_df['time'] = its_df[time_column].dt.to_period(time_conversion.get(time_unit, 'D')).astype(str)
    its_df['time'] = pd.to_datetime(its_df['time']).rank(method='first').astype(int)
    
    its_df = pd.merge(its_df, df[[time_column, outcome_column]], how='left', on=time_column)

    # Set the intervention point
    if intervention_point is not None:
        if isinstance(intervention_point, str) or isinstance(intervention_point, pd.Timestamp):
            intervention_point = its_df[its_df[time_column] >= pd.to_datetime(intervention_point)].iloc[0]['time']

    # Set the split points
    if split_point is not None:
        if isinstance(split_point, (tuple, list)) and len(split_point) == 2:
            start_gap = pd.to_datetime(split_point[0])
            end_gap = pd.to_datetime(split_point[1])
        else:
            raise ValueError("split_point must be a tuple or list with two elements: (start_gap, end_gap)")
    
    # Split the data into pre-split and post-split groups
    df_pre_split = its_df[its_df[time_column] <= start_gap].copy()
    counterfactual_split = its_df[its_df[time_column] > start_gap].copy()
    df_post_split = its_df[its_df[time_column] >= end_gap].copy()
    
    # Perform ITS on the pre-split data
    df_pre_split['intervention'] = (df_pre_split['time'] >= intervention_point).astype(int) if intervention_point else 0
    df_pre_split['time_after_intervention'] = df_pre_split['time'] - df_pre_split['time'][df_pre_split['intervention'] == 1].min()
    df_pre_split['time_after_intervention'] = df_pre_split['time_after_intervention'].apply(lambda x: x if x >= 0 else 0)

    # Create design matrix for pre-split data
    X_pre_split = sm.add_constant(df_pre_split[['time', 'intervention', 'time_after_intervention']])
    
    # Include control variables if provided
    if control_columns:
        X_pre_split = sm.add_constant(df_pre_split[['time', 'intervention', 'time_after_intervention'] + control_columns])
    
    # Fit the model on pre-split data
    model_pre_split = sm.OLS(df_pre_split[outcome_column], X_pre_split)
    results_pre_split = model_pre_split.fit()
    
    if show_summary:
        print("ITS Analysis for Pre-Split Data:")
        print(results_pre_split.summary())
    
    # Calculate the predicted values for the pre-split data
    df_pre_split['predicted'] = results_pre_split.predict(X_pre_split)
    
    # Predict the counterfactual trend for the post-split data using pre-split model
    if counterfactual_postanalysis:
        # Prepare the counterfactual data
        counterfactual_split['intervention'] = 1  # No intervention is considered in the counterfactual scenario
        counterfactual_split['time_after_intervention'] = counterfactual_split['time'] - intervention_point
        counterfactual_split['time_after_intervention'] = counterfactual_split['time_after_intervention'].apply(lambda x: x if x >= 0 else 0)
        
        b0 = results_pre_split.params['const']
        b1 = results_pre_split.params['time']
        b2 = results_pre_split.params['intervention']
        b3 = results_pre_split.params['time_after_intervention']

        counterfactual_split['counterfactual'] = b0 + (b1 * counterfactual_split['time']) + (b2) + (b3 * counterfactual_split['time_after_intervention'])
    
    if not df_post_split.empty:
        
        # Rank the time column to ensure it's in an integer format
        df_post_split['time'] = pd.to_datetime(df_post_split['time']).rank(method='first').astype(int)
        df_post_split[outcome_column].fillna(0, inplace=True)
        X_post_split = sm.add_constant(df_post_split[['time']])
        
        # Fit the linear regression model
        model_post_split = sm.OLS(df_post_split[outcome_column], X_post_split).fit()
                
        # Generate the predicted values from the linear regression model
        df_post_split['predicted_post'] = model_post_split.fittedvalues
        df_post_split['time'] = df_post_split[time_column].apply(lambda x: (x.date() - min_time.date()).days)
    
    if plot:
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the actual outcome as a scatter plot with grey color and 50% opacity
        ax.scatter(its_df['time'], its_df[outcome_column], label='Outcome', color='grey', alpha=0.5, s=point_size)
        
        if intervention_point:
            ax.axvline(x=its_df['time'][its_df['time'] == intervention_point].iloc[0], color='red', linestyle='--', label='Intervention Point', linewidth=line_width)
        
        # Plot the fitted values from the pre-split data
        ax.plot(df_pre_split['time'], df_pre_split['predicted'], label='Fitted values (Pre-Split)', color='blue', linewidth=line_width)
        
        if counterfactual_postanalysis and not df_post_split.empty:
            # Plot the counterfactual line for post-split data
            ax.plot(counterfactual_split['time'], counterfactual_split['counterfactual'], label='Counterfactual (Post-Analysis)', color='green', linestyle='--', linewidth=line_width)
        
        if not df_post_split.empty:
            # Plot the post-split regression line
            ax.plot(df_post_split['time'], df_post_split['predicted_post'], label='Fitted values (Post-Split)', color='purple', linewidth=line_width)
        
        ax.set_ylim(y_lim)
        
        # Set axis labels and title with specified font sizes
        ax.set_xlabel(time_unit, fontsize=axis_label_size)
        if first_col:
            ax.set_ylabel(outcome_column, fontsize=axis_label_size)
        ax.set_title(title, fontsize=title_size)
        
        # Set the size of the tick labels
        ax.tick_params(axis='x', labelsize=tick_label_size)
        ax.tick_params(axis='y', labelsize=tick_label_size)
        
        ax.grid(True)


    return results_pre_split







from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import warnings

def encode_categorical_columns(df, columns):
    """Encode categorical columns with label encoding."""
    label_encoders = {}
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Ensure all data is string before encoding
        label_encoders[col] = le
    return df, label_encoders

def random_forest(df_input, outcome_column, outcome_choice, predictors, reference_group=None, test_size=0.2, random_state=42):
    
    df = df_input.copy()
    df = df[predictors + [outcome_column]]

    table_name = f"random_forest_{outcome_choice}_vs_others"
    print(f"Random Forest analysis for {outcome_choice} as outcome")

    # Encode the outcome column
    le_outcome = LabelEncoder()
    df['outcome'] = le_outcome.fit_transform(df[outcome_column].astype(str))

    # Print classes in the outcome for reference
    print(f"Outcome encoding: {dict(zip(le_outcome.classes_, le_outcome.transform(le_outcome.classes_)))}")

    # Ensure predictors are numeric and handle categorical data
    categorical_predictors = df[predictors].select_dtypes(include=['object', 'category']).columns.tolist()

    # Handling reference group before encoding
    if reference_group:
        for predictor, ref_value in reference_group.items():
            if predictor in categorical_predictors:
                df[predictor] = pd.Categorical(df[predictor], categories=[ref_value] + [x for x in df[predictor].unique() if x != ref_value], ordered=True)     

    # Encode categorical predictors
    df, _ = encode_categorical_columns(df, categorical_predictors)

    X = df.drop(columns=[outcome_column, 'outcome'])
    y = df['outcome']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize and train the Random Forest model
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC AUC Score: {roc_auc:.3f}")

    # Feature importance
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    print("\nFeature Importance:")
    print(feature_importance)
    
    # Optionally save the results
    # add_to_docx(feature_importance, table_name, output_dir='output/analyse')

# Example usage
# random_forest_analysis(df_logis, outcome_column, outcome_choice, predictors, reference_group)

    
    # Optionally save the results
    # add_to_docx(feature_importance, table_name, output_dir='output/analyse')

# Example usage
# random_forest_analysis(df_logis, outcome_column, outcome_choice, predictors, reference_group)
