"""Same selected BigMart model, written as small teaching functions.
Put the train/test CSVs beside this file (or inside data/), then run it.
Requires: numpy, pandas. See the notebook for explanations and validation.
"""
from pathlib import Path
import numpy as np
import pandas as pd

TARGET = 'Item_Outlet_Sales'
KEYS = ['Item_Identifier', 'Outlet_Identifier']


def prepare(data):
    data = data.copy().reset_index(drop=True)
    prices = np.arange(40, 420) * 0.6658
    gaps = (data.Item_MRP.to_numpy()[:, None] - prices) * 10
    errors = np.abs(gaps - np.round(gaps))
    data['Price'] = prices[errors.argmin(axis=1)]
    assert errors.min(axis=1).max() < 1e-7, 'Unexpected price pattern'
    assert (data.Price - data.Item_MRP).abs().max() <= 2.000001
    data['Code'] = data.Item_Identifier.str[3:].astype(int) - 1
    data['Fat'] = data.Item_Fat_Content.replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})
    data.loc[data.Item_Identifier.str.startswith('NC'), 'Fat'] = 'Non Edible'
    return data


def outlet_demand(data):
    units = data[TARGET] / data.Price
    stores = data.assign(Units=units).groupby('Outlet_Identifier').Units.agg(['sum', 'count'])
    types = units.groupby(data.Outlet_Type).mean()
    labels = data.drop_duplicates('Outlet_Identifier').set_index('Outlet_Identifier').Outlet_Type
    demand = (stores['sum'] + 50 * labels.map(types)) / (stores['count'] + 50)
    return demand


def product_curve(data, residual, weights):
    matrix = np.eye(60)[data.Code]                  # One column per product-code number
    differences = np.diff(np.eye(60), n=2, axis=0) # Changes in the curve's slope
    penalty = 20 * np.eye(60) + 150 * differences.T @ differences
    left = matrix.T @ (matrix * weights[:, None]) + penalty
    right = matrix.T @ (residual * weights)
    return np.linalg.solve(left, right)


def fit_model(data):
    demand = outlet_demand(data)
    base = data.Outlet_Identifier.map(demand).to_numpy()
    residual = data[TARGET].to_numpy() / (data.Price.to_numpy() * base) - 1
    weights = (data.Price.to_numpy() / 140)**2 * (base / 16)**2
    curve = product_curve(data, residual, weights)
    remaining = residual - curve[data.Code]
    totals = pd.DataFrame({'Fat': data.Fat, 'Error': remaining * weights,
                           'Weight': weights}).groupby('Fat').sum()
    fat_effect = totals.Error / (totals.Weight + 500)
    return demand, curve, fat_effect


def predict(data, model):
    demand, curve, fat_effect = model
    base = data.Outlet_Identifier.map(demand).to_numpy()
    assert np.isfinite(base).all(), 'This teaching version expects previously seen outlets'
    adjustment = curve[data.Code] + data.Fat.map(fat_effect).fillna(0).to_numpy()
    return np.maximum(data.Price.to_numpy() * base * (1 + adjustment), 0)


def find_data(filename):
    for folder in [Path(__file__).parent, Path.cwd(), Path.cwd() / 'data', Path.home() / 'Downloads']:
        if (folder / filename).exists():
            return folder / filename
    raise FileNotFoundError(f'Place {filename} beside this file or in data/.')


def main():
    train = prepare(pd.read_csv(find_data('train_v9rqX0R.csv')))
    test = prepare(pd.read_csv(find_data('test_AbJTz2l.csv')))
    submission = test[KEYS].copy()
    submission[TARGET] = predict(test, fit_model(train))
    output = Path(__file__).with_name('submission_simple.csv')
    submission.to_csv(output, index=False, float_format='%.8f')
    print(f'Saved {output.name}: {len(submission)} predictions')


if __name__ == '__main__':
    main()
