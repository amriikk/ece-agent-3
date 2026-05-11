import pandas as pd
from operators import OPERATORS

def execute(df, plan):
    """
    Execute a plan against a DataFrame.
    
    Args:
        df:   the original dataset (pandas DataFrame)
        plan: dict with a "steps" list, each step has "op" + parameters
    
    Returns:
        result: final DataFrame after all steps
        trace:  list of dicts showing table shape at each step
    """
    trace = []
    current = df.copy()

    for i, step in enumerate(plan["steps"]):
        op_name = step["op"]
        
        # Validate the operator exists
        if op_name not in OPERATORS:
            raise ValueError(f"Step {i+1}: Unknown operator '{op_name}'")
        
        # Strip "op" key — remaining keys are the operator's parameters
        params = {k: v for k, v in step.items() if k != "op"}
        
        input_rows  = len(current)
        input_cols  = list(current.columns)
        
        # Execute
        current = OPERATORS[op_name](current, **params)
        
        # Record trace
        trace.append({
            "step":        i + 1,
            "op":          op_name,
            "input_rows":  input_rows,
            "output_rows": len(current),
            "output_cols": list(current.columns),
        })
        
        # Safety check — if result is empty, warn but keep going
        if len(current) == 0:
            print(f"  ⚠️  Warning: Step {i+1} ({op_name}) produced an empty table")

    return current, trace

def format_trace(trace):
    """
    Pretty-print the execution trace as a table.
    Matches the format required by the assignment.
    """
    header = f"{'Step':<6} {'Operation':<25} {'Input Rows':<14} {'Output Rows':<12}"
    divider = "-" * len(header)
    lines = [header, divider]
    
    for t in trace:
        lines.append(
            f"{t['step']:<6} {t['op']:<25} {t['input_rows']:<14} {t['output_rows']:<12}"
        )
    
    return "\n".join(lines)