# experiment_tools
Python の機械学習実験における自動 log 記録のシステムテンプレート.

## 概要
機械学習実験をする際に, 学習した結果を保存する必要がある.  
そのための log ファイルを自動で生成するシステムを提供する.

## ファイルの説明
- `__init__.py`
    - 空ファイル, モジュールとして呼び出す上で必要
- `set_random_seed.py`
    - 各種機械学習フレームワークの乱数シードを固定する関数を管理する
        - `fix_seed()` 関数
- `set_up.py`
    - 実験者の環境を log ファイルに記録し, 実験を開始する
        - `get_directory()` 関数
        - `get_git_info()` 関数
        - `get_os_info()` 関数
        - `get_pip_list()` 関数
        - `start_experiment()` 関数
- `start_logging.py`
    - log を初期化する
        - `get_logger()` 関数