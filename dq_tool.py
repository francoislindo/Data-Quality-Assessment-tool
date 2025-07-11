"""
dq_tool.py
A command-line tool for basic data quality assessment on CSV/Excel files.
"""

import argparse
import json
import os
from typing import Any, Dict
import pandas as pd
from pandas.api import types as pdt
import re
import streamlit as st
import llm_advisor


def check_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes completeness metrics for each column in the DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with columns: column, total_rows, null_count, completeness_pct.
    """
    total_rows = len(df)
    null_counts = df.isna().sum()
    completeness = []
    for col in df.columns:
        null_count = null_counts[col]
        non_null = total_rows - null_count
        completeness_pct = 100.0 * non_null / total_rows if total_rows > 0 else 0.0
        completeness.append({
            'column': col,
            'total_rows': total_rows,
            'null_count': int(null_count),
            'completeness_pct': completeness_pct
        })
    return pd.DataFrame(completeness)


def infer_expected_types(df: pd.DataFrame) -> dict[str, str]:
    """
    Infers the expected logical type for each column using pandas.api.types.
    Args:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        dict[str, str]: Mapping of column name to expected type as a string.
    """
    expected_types = {}
    for col in df.columns:
        if pdt.is_integer_dtype(df[col]):
            expected_types[col] = 'int'
        elif pdt.is_float_dtype(df[col]):
            expected_types[col] = 'float'
        elif pdt.is_bool_dtype(df[col]):
            expected_types[col] = 'bool'
        elif pdt.is_datetime64_any_dtype(df[col]):
            expected_types[col] = 'datetime'
        elif pdt.is_string_dtype(df[col]):
            expected_types[col] = 'str'
        else:
            expected_types[col] = 'object'
    return expected_types


def check_type_accuracy(df: pd.DataFrame, expected_types: dict[str, str]) -> pd.DataFrame:
    """
    Checks type accuracy for each column by comparing each value to the expected type.
    Args:
        df (pd.DataFrame): The input DataFrame.
        expected_types (dict[str, str]): Mapping of column name to expected type.
    Returns:
        pd.DataFrame: DataFrame with columns: column, expected_type, mismatched_rows, accuracy_pct.
    """
    type_map = {
        'int': int,
        'float': float,
        'bool': bool,
        'datetime': pd.Timestamp,
        'str': str,
        'object': object
    }
    results = []
    total_rows = len(df)
    for col in df.columns:
        expected_type = expected_types.get(col, 'object')
        py_type = type_map.get(expected_type, object)
        mismatches = 0
        for val in df[col]:
            if pd.isna(val):
                continue  # skip nulls
            if expected_type == 'datetime':
                if not isinstance(val, pd.Timestamp):
                    mismatches += 1
            elif expected_type == 'float':
                # int is also a valid float
                if not (isinstance(val, float) or isinstance(val, int)):
                    mismatches += 1
            elif expected_type == 'int':
                # bool is subclass of int, but we want to exclude bool
                if not (isinstance(val, int) and not isinstance(val, bool)):
                    mismatches += 1
            elif expected_type == 'str':
                if not isinstance(val, str):
                    mismatches += 1
            elif expected_type == 'bool':
                if not isinstance(val, bool):
                    mismatches += 1
            # object: skip strict checking
        matches = total_rows - mismatches
        accuracy_pct = 100.0 * matches / total_rows if total_rows > 0 else 0.0
        results.append({
            'column': col,
            'expected_type': expected_type,
            'mismatched_rows': mismatches,
            'accuracy_pct': accuracy_pct
        })
    return pd.DataFrame(results)


def check_pattern_recognition(df: pd.DataFrame) -> pd.DataFrame:
    """
    For object-dtype columns, detects which pattern (email, phone, postal_code) best matches the data.
    Args:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with columns: column, detected_pattern, match_pct.
    """
    pattern_rules = {
        'email': r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$',
        'phone': r'^\+?\d{10,15}$',
        'postal_code': r'^\d{5}(-\d{4})?$'
    }
    results = []
    for col in df.columns:
        if df[col].dtype == object:
            best_pattern = None
            best_pct = 0.0
            for pattern_name, pattern in pattern_rules.items():
                match_count = 0
                total = 0
                regex = re.compile(pattern)
                for val in df[col]:
                    if pd.isna(val):
                        continue
                    if regex.match(str(val)):
                        match_count += 1
                    total += 1
                pct = 100.0 * match_count / total if total > 0 else 0.0
                if pct > best_pct:
                    best_pct = pct
                    best_pattern = pattern_name
            results.append({
                'column': col,
                'detected_pattern': best_pattern if best_pattern else '',
                'match_pct': best_pct
            })
    return pd.DataFrame(results)


def main() -> None:
    """
    Main function to parse arguments, read input file, run checks, and write report.
    """
    parser = argparse.ArgumentParser(description="Data Quality Assessment Tool")
    parser.add_argument('--input', required=True, help='Path to input CSV or Excel file')
    parser.add_argument('--report', default='dq_report.json', help='Path to output report file (default: dq_report.json)')
    parser.add_argument('--advisor', action='store_true', help='Call LLM advisor to suggest fixes for worst issues')
    args = parser.parse_args()

    input_path = args.input
    report_path = args.report

    _, ext = os.path.splitext(input_path)
    ext = ext.lower()

    if ext == '.csv':
        df = pd.read_csv(input_path, engine="pyarrow")
    else:
        df = pd.read_excel(input_path)

    # Run checks
    completeness_df = check_completeness(df)
    expected_types = infer_expected_types(df)
    type_accuracy_df = check_type_accuracy(df, expected_types)
    pattern_df = check_pattern_recognition(df)

    # Merge on 'column' (outer join)
    merged = completeness_df.merge(type_accuracy_df, on='column', how='outer')
    merged = merged.merge(pattern_df, on='column', how='outer')

    # Convert to plain dict
    report = merged.to_dict(orient='records')

    # Pretty-print
    try:
        from tabulate import tabulate
        print(tabulate(merged, headers='keys', tablefmt='grid', showindex=False))
    except ImportError:
        print(json.dumps(report, indent=2))

    # Ensure report directory exists
    report_dir = os.path.dirname(os.path.abspath(report_path))
    if report_dir and not os.path.exists(report_dir):
        os.makedirs(report_dir, exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    if getattr(args, 'advisor', False):
        llm_advisor.suggest_fixes(report_path)

    # Generate HTML summary report using ydata-profiling
    try:
        from ydata_profiling import ProfileReport
        html_path = os.path.splitext(report_path)[0] + '.html'
        profile = ProfileReport(df, title="Data Quality Profile Report", explorative=True)
        profile.to_file(html_path)
        print(f"HTML summary report saved to: {html_path}")
    except ImportError:
        print("ydata-profiling is not installed. Skipping HTML summary report.")


if __name__ == "__main__":
    main()

st.title("Data Quality Assessment Tool")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:", df.head())

    # Run checks
    completeness_df = check_completeness(df)
    expected_types = infer_expected_types(df)
    type_accuracy_df = check_type_accuracy(df, expected_types)
    pattern_df = check_pattern_recognition(df)

    # Merge results
    merged = completeness_df.merge(type_accuracy_df, on='column', how='outer')
    merged = merged.merge(pattern_df, on='column', how='outer')

    st.write("Data Quality Report")
    st.dataframe(merged) 