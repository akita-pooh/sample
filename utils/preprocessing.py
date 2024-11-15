from typing import Tuple

import lightgbm as lgb
import pandas as pd


def df_index_resetter(df: pd.DataFrame) -> pd.DataFrame:
    """
    データフレームのインデックスをリセットする関数

    Parameters
    ----------
    df: pd.DataFrame
        インデックスをリセットするデータフレーム

    Returns
    ----------
    df: pd.DataFrame
        リセット後のデータフレーム
    """
    # データフレームをリセットする
    df = df.reset_index().drop("index", axis=1)

    return df


def data_split(
        rate: float, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    データフレーム分割する関数

    Parameters
    ----------
    rate: float
        分割する割合
    df: pd.DataFrame
        分割するデータフレーム

    Returns
    ----------
    front: pd.DataFrame
        分割後の前半のデータフレーム
    back: pd.DataFrame
        分割後の後半のデータフレーム
    """
    # 境界値を求める
    boundary = len(df) * rate
    boundary = int(boundary)

    # データを分ける
    front, back = df[:boundary], df[boundary:]
    front, back = df_index_resetter(front), df_index_resetter(back)

    return front, back


def make_datasets(cfg: dict) -> dict:
    """
    実験で使用する config を取得する関数

    Parameters
    ----------
    cfg: dict
        実験に使う値の config データ

    Returns
    ----------
    out: dict
        データセットの辞書
    """
    # データを読み込んで Pandas のデータフレームにする
    df = pd.read_csv(cfg["data_path"])
    pre_df = df.copy()

    # config の lag を参照して入力データを整形
    for i in range(cfg["lag"]):
        tmp_col = f"y_{i + 1}_shift"
        pre_df[tmp_col] = pre_df["y"].shift(i + 1)
    
    # データフレームを学習用とテスト用に分割する
    train_df, test_df = data_split(
        cfg["sampling_rate"]["train"], pre_df
    )

    # 学習データの欠損部分を削除する
    train_df = train_df.dropna(how="any")
    train_df = df_index_resetter(train_df)

    # テスト用データフレームを検証用と評価用に分割する
    valid_df, eval_df = data_split(
        cfg["sampling_rate"]["valid"], test_df
    )

    # 学習データの変数分離
    X_train = train_df.drop("y", axis=1)
    y_train = train_df["y"]

    # 検証データの変数分離
    X_valid = valid_df.drop("y", axis=1)
    y_valid = valid_df["y"]

    # 学習・検証データのデータセットを作成
    train_dataset = lgb.Dataset(X_train, y_train)
    valid_dataset = lgb.Dataset(X_valid, label=y_valid, reference=train_dataset)

    # データを辞書型にまとめる
    out = {
        "org_data": df,
        "train_data": train_df,
        "train_dataset": train_dataset,
        "valid_data": valid_df,
        "valid_dataset": valid_dataset,
        "eval_data": eval_df
    }

    return out
