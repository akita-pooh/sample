# sample
Python の機械学習実験のためのサンプル.

## 概要
機械学習実験のためのフレームワークのサンプルを提供する.

## フォルダの説明
- `config`
    - 実験内容ごとに使う config ファイルを管理する
        - `default`
        - `experiment`
        - `README.md`
- `data`
    - 実験に使うデータを格納する
        - `data_visual.png`
        - `sample_data_downsized.csv`
        - `sample_data_without_noise_downsized.csv`
        - `sample_data_without_noise.csv`
        - `sample_data.csv`
        - `README.md`
- `experiment_tools`
    - Python の機械学習実験における自動 log 記録のシステムテンプレート([参照](https://github.com/akita-pooh/experiment_tools))
        - `log.png`
        - `__init__.py`
        - `set_random_seed.py`
        - `set_up.py`
        - `start_logging.py`
        - `README.md`
- `models`
    - 深層学習モデルのアーキテクチャを作成/管理する
        - `__init__.py`
        - `networks.py`
        - `README.md`
- `outputs`
    - 実験結果を保存する
        - `README.md`
- `scripts`
    - 機械学習実験を実際に行うディレクトリ
        - `train_lgb.py`
        - `train_nn.py`
        - `README.md`
- `trainers`
    - 深層学習モデルの学習に必要なフレームワーク
        - `__init__.py`
        - `loop.py`
        - `opt.py`
        - `README.md`
- `utils`
    - 各モデルでの実験に共通する有用な関数をまとめる
        - `__init__.py`
        - `cfg_diff.py`
        - `preprocessing.py`
        - `result.py`
        - `README.md`

## ファイルの説明
- `.gitignore`
    - GitHub にアップロードしないファイルやフォルダを指定
- `requirements.txt`
    - プロジェクトで必要な Python ライブラリの管理

## LightGBM を使用したことがない人へ
本サンプルでは LightGBM のモデルを利用する.  
これを使うためにはローカル環境で前もって次のコマンドを実行する必要がある.

```
brew install libomp
```

## ローカル環境のセットアップ
仮想環境を構築する.  

```
python3 -m venv venv
source ./venv/bin/activate
```

`pip3` を使用する場合, リポジトリのターミナル上で以下のコマンドを実行する.  

```
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

`PYTHONPATH` を通して追加する.  

```
echo 'export PYTHONPATH=..' >> ~/.bashrc
source ~/.bashrc
```

## 実行方法
`scripts` ディレクトリに移動して, 実行したいファイルを選択する.  
他のディレクトリで実行すると失敗する.

## Commit ルール
Commit の際は以下のルールに従って意味のある変更ごとに行う.  

🎉 初めてのコミット (Initial Commit)  
🔖 バージョンタグ (Version Tag)  
✨ 新機能 (New Feature)  
🐛 バグ修正 (Bugfix)  
♻️ リファクタリング (Refactoring)  
📚 ドキュメント (Documentation)  
🎨 デザインUI/UX (Accessibility)  
🐎 パフォーマンス (Performance)  
🔧 ツール (Tooling)  
🚨 テスト (Tests)  
💩 非推奨追加 (Deprecation)  
🗑️ 削除 (Removal)  
🚧 WIP (Work In Progress)  