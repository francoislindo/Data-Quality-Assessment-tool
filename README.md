# Data Quality Assessment Tool

## Purpose

`dq_tool.py` is a command-line tool for performing basic data quality assessment on CSV and Excel files. It checks for completeness, type accuracy, and recognizes common data patterns (such as email, phone, and postal codes) in your datasets. The tool generates a comprehensive report in JSON format and can pretty-print results in the terminal.

## Setup

1. **Clone or download this repository.**
2. **Create and activate a virtual environment (recommended):**
   ```sh
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```sh
   pip install pandas pyarrow tabulate
   ```
   - `tabulate` is optional, but recommended for pretty terminal output.

## Usage

Run the tool from the command line, specifying your input file:

```sh
python dq_tool.py --input sample.csv
```

- Use `--input` to specify the path to your CSV or Excel file.
- Optionally, use `--report` to specify the output report file (default: `dq_report.json`).

Example:
```sh
python dq_tool.py --input data.xlsx --report results/my_report.json
```

## Extending Pattern Rules

To add or modify pattern recognition rules (e.g., to detect new data types):

1. Open `dq_tool.py` and locate the `check_pattern_recognition` function.
2. Update the `pattern_rules` dictionary to add new rules or adjust existing ones. For example:
   ```python
   pattern_rules = {
       'email': r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$',
       'phone': r'^\\+?\\d{10,15}$',
       'postal_code': r'^\\d{5}(-\\d{4})?$',
       'custom_rule': r'^your-regex-here$'  # Add your own
   }
   ```
3. Save the file and re-run the tool. The new pattern will be automatically included in the analysis.

---

For questions or contributions, please open an issue or pull request. 