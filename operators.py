import pandas as pd
import numpy as np

def get_operator(name):
    return OPERATORS[name]  #OPERATORS at the bottom

def derive_columns(df, derive):
    df = df.copy()
    for spec in derive:
        new_col = spec["new_column"]
        dtype = spec["type"]

        if dtype == "arithmetic":
            left = _resolve_operand(df, spec["left"])
            right = _resolve_operand(df, spec["right"])
            op = spec["operation"]
            if op == "add":
                df[new_col] = left + right
            elif op == "subtract":
                df[new_col] = left - right
            elif op == "multiply":
                df[new_col] = left * right
            elif op == "divide":
                df[new_col] = left / right.replace(0, np.nan)

        elif dtype == "date_diff":
            # e.g. days between two date columns
            col_a = pd.to_datetime(df[spec["start"]], errors="coerce")
            col_b = pd.to_datetime(df[spec["end"]], errors="coerce")
            df[new_col] = (col_b - col_a).dt.days

        elif dtype == "extract_date_part":
            # e.g. extract year, month from a date column
            col = pd.to_datetime(df[spec["column"]], errors="coerce")
            part = spec["part"]  # "year", "month", "day", "dayofweek"
            df[new_col] = getattr(col.dt, part)

    return df


def _resolve_operand(df, operand):
    """Helper: operand is either {"type": "column", "value": "col_name"}
                               or {"type": "literal", "value": 123}"""
    if operand["type"] == "column":
        return pd.to_numeric(df[operand["value"]], errors="coerce")
    elif operand["type"] == "literal":
        return operand["value"]
    
def filter_rows(df, conditions):
    df = df.copy()
    mask = pd.Series([True] * len(df), index=df.index)
    
    for cond in conditions:
        col = df[cond["column"]]
        val = cond["value"]
        op  = cond["operator"]
        
        # coerce numeric columns for comparison
        if op in [">", "<", ">=", "<="]:
            col = pd.to_numeric(col, errors="coerce")
            val = float(val)
        
        if op == ">":
            mask &= col > val
        elif op == "<":
            mask &= col < val
        elif op == ">=":
            mask &= col >= val
        elif op == "<=":
            mask &= col <= val
        elif op == "==":
            mask &= col == val
        elif op == "!=":
            mask &= col != val
        elif op == "contains":
            mask &= col.astype(str).str.contains(val, case=False, na=False)
    
    return df[mask]

def group_and_aggregate(df, group_by, metrics):
    agg_dict = {}
    rename_dict = {}
    
    for m in metrics:
        col = m["column"]
        func = m["function"]
        alias = m["as"]
        
        # ensure numeric
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
        if col not in agg_dict:
            agg_dict[col] = []
        agg_dict[col].append(func)
        rename_dict[(col, func)] = alias
    
    result = df.groupby(group_by).agg(agg_dict).reset_index()
    result.columns = [
        rename_dict.get(col, col[0] if isinstance(col, tuple) else col)
        for col in result.columns
    ]
    return result

def sort_rows(df, sort_by):
    columns = [s["column"] for s in sort_by]
    ascending = [s["direction"] == "asc" for s in sort_by]
    return df.sort_values(by=columns, ascending=ascending).reset_index(drop=True)

def limit_rows(df, k):
    return df.head(k).reset_index(drop=True)

def select_columns(df, columns):
    return df[columns]

def distinct_rows(df, columns=None):
    return df.drop_duplicates(subset=columns).reset_index(drop=True)

OPERATORS = {
    "derive_columns":      derive_columns,
    "filter_rows":         filter_rows,
    "group_and_aggregate": group_and_aggregate,
    "sort_rows":           sort_rows,
    "limit_rows":          limit_rows,
    "select_columns":      select_columns,
    "distinct_rows":       distinct_rows,
}