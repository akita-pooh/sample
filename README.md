# sample
Python の機械学習実験のためのサンプル.

## 概要
機械学習実験のためのフレームワークのサンプルを提供する.

## フォルダの説明


## ファイルの説明


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
echo 'export PYTHONPATH=../..' >> ~/.bashrc
source ~/.bashrc
```

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