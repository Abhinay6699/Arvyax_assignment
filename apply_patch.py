import re

with open('new_functions.py', 'r', encoding='utf-8') as f:
    patch = f.read()

with open('main.py', 'r', encoding='utf-8') as f:
    original = f.read()

pattern = re.compile(r"def run_error_analysis\(train_df, X_combined_train, model_results, top_n=15\):.*?def print_summary", re.DOTALL)
new_content = pattern.sub(patch + "\n\ndef print_summary", original)

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
