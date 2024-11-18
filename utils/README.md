# utils
各モデルでの実験に共通する有用な関数をまとめる.

## 概要
各実験において, 前処理や評価などの共通するワークフローを提供する。

## ファイルの説明
- `__init__.py`
    - 空ファイル, モジュールとして呼び出す上で必要
- `cfg_diff.py`
    - config の差分を取得する
        - `get_config()` 関数
        - `get_diff()` 関数
- `preprocessing.py`
    - 各モデルにエンコーディングするためのデータの前処理を行う
        - `df_index_resetter()` 関数
        - `data_split()` 関数
        - `make_datasets()` 関数
        - `to_torch()` 関数
        - `make_datasets_for_nn()` 関数
- `result.py`
    - モデルの学習後に行う評価等の処理
        - `plot_data()` 関数
        - `extract_feature_importance()` 関数
        - `predict()` 関数