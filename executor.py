def execute(df, plan):
    trace = []
    current = df.copy()
    for step in plan["steps"]:
        op_name = step["op"]
        params = {k: v for k, v in step.items() if k != "op"}
        input_rows = len(current)
        current = OPERATORS[op_name](current, **params)
        trace.append({
            "op": op_name,
            "input_rows": input_rows,
            "output_rows": len(current)
        })
    return current, trace