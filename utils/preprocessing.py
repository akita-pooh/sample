from typing import Literal, Tuple

import lightgbm as lgb
import pandas as pd
import torch
from torch.utils.data import TensorDataset


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
    LightGBMの実験で使用するデータセットを作成する関数

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


def to_torch(
        X: pd.DataFrame,
        y: pd.DataFrame,
        torch_type: Literal["Float", "Long"] = "Float",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    データを Tensor 型に変換するする関数

    Parameters
    ----------
    X: pd.DataFrame
        説明変数のデータフレーム
    y: pd.DataFrame
        目的変数のデータフレーム
    torch_type: Literal["Float", "Long"] = "Float"
        PyTorch で計算できる型

    Returns
    ----------
    X_data: torch.Tensor
        Tensor 型に変換した説明変数のデータ
    y_data: torch.Tensor
        Tensor 型に変換した目的変数のデータ
    """
    # Tensor 型に変換するために一旦 NumPy に直す
    X_data = X.to_numpy()
    y_data = y.to_numpy()

    # PyTorch で計算できる型に変換する
    if torch_type == "Float":
        X_data = torch.from_numpy(X_data).float()
        y_data = torch.from_numpy(y_data).float()
    else:
        X_data = torch.from_numpy(X_data).long()
        y_data = torch.from_numpy(y_data).long()

    return X_data, y_data


def make_datasets_for_nn(cfg: dict) -> dict:
    """
    ニューラルネットワーク系の実験で使用するデータセットを作成する関数

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
    X_train, y_train = to_torch(X_train, y_train, torch_type=cfg["torch_type"])
    X_valid, y_valid = to_torch(X_valid, y_valid, torch_type=cfg["torch_type"])
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)

    out = {
        "org_data": df,
        "train_data": train_df,
        "train_dataset": train_dataset,
        "valid_data": valid_df,
        "valid_dataset": valid_dataset,
        "eval_data": eval_df
    }

    return out
