import os
import pandas as pd
from sklearn.metrics import f1_score


def evaluate_predictions(df, target_col, gold_col='label'):
    if gold_col not in df.columns or target_col not in df.columns:
        return None
    
    df.dropna(subset=[target_col, gold_col], inplace=True)

    y_true = df[gold_col].astype(int)
    y_pred = df[target_col].astype(int)

    return {
        'f1_micro': round(f1_score(y_true, y_pred, average='micro'), 4),
        'f1_macro': round(f1_score(y_true, y_pred, average='macro'), 4),
    }


def process_directory(directory):
    results = []

    for filename in os.listdir(directory):
        if filename.endswith("<model>-<lang>.csv"):
            filepath = os.path.join(directory, filename)
            source_lang = filename.split("_")[-1].replace(".csv", "")

            df = pd.read_csv(filepath)
            gold_col = "label" if "label" in df.columns else "gold_label"

            for col in df.columns:
                if col == gold_col or col == 'text' or col == 'id':
                    continue

                scores = evaluate_predictions(df, col, gold_col=gold_col)
                if scores:
                    results.append({
                        'file': filename,
                        'source_lang': source_lang,
                        'target_lang': col,
                        **scores
                    })

    return pd.DataFrame(results)

directory_path = "runs/"
results_df = process_directory(directory_path)
print(results_df)

