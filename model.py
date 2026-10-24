"""BigMart sales prediction using standard Ridge regression."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error

TARGET = 'Item_Outlet_Sales'
KEYS = ['Item_Identifier', 'Outlet_Identifier']


def make_features(data, prices):
    categories = data[['Outlet_Identifier']].copy()
    categories['Product_Code'] = data.Item_Identifier.str[3:]
    categories['Fat'] = data.Item_Fat_Content.replace(
        {'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})
    categories.loc[data.Item_Identifier.str.startswith('NC'), 'Fat'] = 'Non Edible'
    encoded = pd.get_dummies(categories, dtype=float)
    price = data.Item_Identifier.map(prices).fillna(data.Item_MRP)
    return encoded.mul(price, axis=0)


def rmse(actual, predicted):
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def evaluate(train, test):
    # Price averages use input columns only, never test sales.
    prices = pd.concat([train, test]).groupby('Item_Identifier').Item_MRP.mean()
    X = make_features(train, prices)
    y = train[TARGET].to_numpy()
    dev, hold = train_test_split(np.arange(len(y)), test_size=0.2,
                                random_state=2027, stratify=train.Outlet_Identifier)
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    for alpha in [1000, 10000, 100000, 1000000, 10000000]:
        scores = []
        for fitting, checking in folds.split(dev):
            model = Ridge(alpha=alpha, fit_intercept=False)
            model.fit(X.iloc[dev[fitting]], y[dev[fitting]])
            predicted = np.maximum(model.predict(X.iloc[dev[checking]]), 0)
            scores.append(rmse(y[dev[checking]], predicted))
        results.append({'alpha': alpha, 'mean_cv_rmse': float(np.mean(scores))})
    best_alpha = min(results, key=lambda row: row['mean_cv_rmse'])['alpha']
    model = Ridge(alpha=best_alpha, fit_intercept=False).fit(X.iloc[dev], y[dev])
    predicted = np.maximum(model.predict(X.iloc[hold]), 0)
    report = {'selected_alpha': best_alpha, 'development_rows': len(dev),
              'holdout_rows': len(hold), 'cv_results': results,
              'holdout_rmse': rmse(y[hold], predicted)}
    return X, y, prices, dev, hold, predicted, report


def main():
    folder = Path(__file__).resolve().parent
    train = pd.read_csv(folder / 'data/train.csv')
    test = pd.read_csv(folder / 'data/test.csv')
    sample = pd.read_csv(folder / 'data/sample_submission.csv')
    X, y, prices, dev, hold, predicted, report = evaluate(train, test)
    model = Ridge(alpha=report['selected_alpha'], fit_intercept=False).fit(X, y)
    test_features = make_features(test, prices).reindex(columns=X.columns, fill_value=0)
    submission = test[KEYS].copy()
    submission[TARGET] = np.maximum(model.predict(test_features), 0)
    assert submission[KEYS].equals(sample[KEYS])
    assert submission.columns.tolist() == sample.columns.tolist()
    assert np.isfinite(submission[TARGET]).all()
    submission.to_csv(folder / 'submission.csv', index=False, float_format='%.8f')
    (folder / 'validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print('Saved submission.csv with', len(submission), 'rows')


if __name__ == '__main__':
    main()
