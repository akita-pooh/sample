import difflib
import json
import logging
from typing import Tuple


def get_config(defaut: str, experiment: str) -> Tuple[dict, dict, str, str]:
    """
    実験で使用する config を取得する関数

    Parameters
    ----------
    defaut: str
        デフォルト config のパス
    experiment: str
        実験用 config のパス

    Returns
    ----------
    default_cfg: dict
        デフォルト値の config データ
    exp_cfg: dict
        実験に使う値の config データ
    default_cfg_str: str
        デフォルト値の config データを文字列としたもの
    exp_cfg_str: str
        実験に使う値の config データを文字列としたもの
    """
    # 各 config ファイルを読み込み, 辞書型のデータとする
    with open(defaut) as f:
        default_cfg = json.load(f)
    with open(experiment) as f:
        exp_cfg = json.load(f)

    # 各辞書型 config データを文字列データに変換する
    default_cfg_str = json.dumps(default_cfg, indent=4)
    exp_cfg_str = json.dumps(exp_cfg, indent=4)

    return default_cfg, exp_cfg, default_cfg_str, exp_cfg_str


def get_diff(default: str, exp: str, logger: logging.Logger) -> logging.Logger:
    """
    文字列の config データを比較し, 差分を取得する関数

    Parameters
    ----------
    defaut: str
        デフォルト値の config データを文字列としたもの
    exp: str
        実験に使う値の config データを文字列としたもの
    logger: logging.Logger
        実験の結果を記録する log データ

    Returns
    ----------
    logger: logging.Logger
        実験の結果を記録する log データ
    """
    # デフォルト値と比較して差分を得る
    differ = difflib.ndiff(
        default.splitlines(keepends=True), exp.splitlines(keepends=True)
    )

    # 差分をまとめて1つの文字列にする
    diff_parts = "\n"
    for line in differ:
        if line.startswith("+") or line.startswith("-"):
            diff_parts += "\n" + line.strip()
    diff_parts += "\n"

    # log に記録する
    logger.info(f"diff: {diff_parts}")

    return logger