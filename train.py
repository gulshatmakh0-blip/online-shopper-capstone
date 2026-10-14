"""Reproducible training. Run: python train.py"""
from pathlib import Path
import json
import math
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent
CAT = ['Month', 'VisitorType', 'Weekend', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']

def load_data():
    raw = pd.read_csv(ROOT / 'data/online_shoppers_intention.csv')
    return raw, raw.drop_duplicates().reset_index(drop=True)

def make_pipeline(model, X):
    numeric = [c for c in X.columns if c not in CAT]
    preprocessing = ColumnTransformer([
        ('num', Pipeline([('fill', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric),
        ('cat', Pipeline([('fill', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore'))]), CAT)
    ])
    return Pipeline([('prepare', preprocessing), ('model', model)])

def run():
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    raw, df = load_data()
    y = df.Revenue.astype(int)
    X = df.drop(columns=['Revenue', 'PageValues'])
    # Remove duplicate records BEFORE splitting. Keep final test out of tuning.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    searches = {
        'Logistic regression': GridSearchCV(make_pipeline(LogisticRegression(max_iter=2000, random_state=42), X), {'model__C': [.1, 1, 10]}, scoring='average_precision', cv=cv, n_jobs=1),
        'Random forest': GridSearchCV(make_pipeline(RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1), X), {'model__max_depth': [8, None], 'model__min_samples_leaf': [3, 10]}, scoring='average_precision', cv=cv, n_jobs=1)
    }
    fitted = {'Baseline': make_pipeline(DummyClassifier(strategy='prior'), X).fit(X_train, y_train)}
    cv_scores = {}
    params = {}
    for name, search in searches.items():
        print('Training:', name, flush=True)
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        cv_scores[name] = float(search.best_score_)
        params[name] = search.best_params_
    winner = max(cv_scores, key=cv_scores.get)
    rows = []
    for name, model in fitted.items():
        p = model.predict_proba(X_test)[:, 1]
        pred = p >= .5
        k = math.ceil(.2 * len(y_test))
        # Baseline scores are tied: expected random selection precision equals prevalence.
        precision_top = float(y_test.mean()) if name == 'Baseline' else float(y_test.iloc[np.argsort(-p, kind='stable')[:k]].mean())
        rows.append({'Model': name, 'CV_PR_AUC': cv_scores.get(name, np.nan), 'Test_PR_AUC': average_precision_score(y_test, p), 'ROC_AUC': roc_auc_score(y_test, p), 'Precision_0.5': precision_score(y_test, pred, zero_division=0), 'Recall_0.5': recall_score(y_test, pred), 'F1_0.5': f1_score(y_test, pred), 'Precision_top20': precision_top, 'Lift_top20': precision_top / y_test.mean()})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / 'metrics.csv', index=False)
    joblib.dump(fitted[winner], out / 'model.joblib')
    p = fitted[winner].predict_proba(X_test)[:, 1]
    pd.DataFrame({'actual': y_test, 'score': p}).to_csv(out / 'test_predictions.csv', index=False)
    X_test.head(20).to_csv(ROOT / 'data/demo_sessions.csv', index=False)
    summary = {'raw_rows': len(raw), 'rows': len(df), 'duplicates_removed': len(raw)-len(df), 'missing_cells': int(raw.isna().sum().sum()), 'purchase_rate': float(y.mean()), 'train_rows': len(X_train), 'test_rows': len(X_test), 'test_purchase_rate': float(y_test.mean()), 'winner': winner, 'best_parameters': params, 'seed': 42, 'excluded': ['Revenue (target)', 'PageValues (potential leakage)']}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(['No purchase', 'Purchase'], y.value_counts().reindex([0, 1]), color=['#94a3b8', '#0d9488'])
    ax.set(ylabel='Sessions', title='Purchase class imbalance')
    fig.tight_layout(); fig.savefig(out/'classes.png', dpi=150); plt.close(fig)
    for col, fname in [('VisitorType', 'visitors'), ('Month', 'months')]:
        tab = df.groupby(col).Revenue.agg(['mean', 'count'])
        tab.to_csv(out / (fname+'.csv'))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(tab.index.astype(str), tab['mean']*100, color='#0d9488')
        ax.set(ylabel='Purchase rate (%)', title=f'Purchase rate by {col}')
        ax.tick_params(axis='x', rotation=25)
        fig.tight_layout(); fig.savefig(out/(fname+'.png'), dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([df.loc[y==0, 'ProductRelated'], df.loc[y==1, 'ProductRelated']], tick_labels=['No purchase', 'Purchase'], showfliers=False)
    ax.set(ylabel='Product pages viewed', title='Product page views by purchase outcome')
    fig.tight_layout(); fig.savefig(out/'pages.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, p >= .5, display_labels=['No purchase', 'Purchase'], ax=ax, colorbar=False)
    ax.set_title('Final test, threshold 0.5')
    fig.tight_layout(); fig.savefig(out/'confusion.png', dpi=150); plt.close(fig)
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))
    return metrics, summary

if __name__ == '__main__':
    run()
