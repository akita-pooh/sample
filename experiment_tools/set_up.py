import datetime
import logging
import os
import platform
import subprocess

from experiment_tools.set_random_seed import fix_seed
from experiment_tools.start_logging import get_logger


def get_directory(cfg: dict, date_time: datetime.datetime) -> dict:
    """
    実験結果を保存するディレクトリを取得する関数

    Parameters
    ----------
    cfg: dict
        実験で参照する config データ
    date_time: datetime.datetime
        実験を開始した時刻のインスタンス

    Returns
    ----------
    cfg: dict
        上書きされた実験で参照する config データ
    """
    # 実験結果の出力先のディレクトリ名を指定
    output_dir = "../../outputs"
    model_output_dir = output_dir + "/" + cfg["model_type"]

    # config 内の "model_type" の値が存在しない場合はディレクトリを新たに作成する
    if cfg["model_type"] not in os.listdir(output_dir):
        os.mkdir(model_output_dir)

    # 同日に同じモデルで実験を行っていない場合は日付のディレクトリを新たに作成する
    date = str(date_time.date())
    date_dir = model_output_dir + "/" + date
    if date not in os.listdir(model_output_dir):
        os.mkdir(date_dir)

    # 実験開始時刻のディレクトリを作成する
    time = str(date_time.time()).split(".")[0].replace(":", "-")
    time_dir = date_dir + "/" + time
    os.mkdir(time_dir)

    # config データの出力先のディレクトリを上書きする
    cfg["log"]["log_file"] = time_dir + "/" + cfg["log"]["log_file"]

    return cfg


def get_git_info(logger: logging.Logger) -> logging.Logger:
    """
    Gitの情報を取得し, 実験者環境を log ファイルに記録する関数

    Parameters
    ----------
    logger: logging.Logger
        実験の結果を記録する log データ

    Returns
    ----------
    logger: logging.Logger
        実験の結果を記録する log データ
    """
    # commit id を取得する
    git_info = "commit id: "
    git_info += subprocess.check_output(
        ["git", "rev-parse", "HEAD"]
    ).decode().strip()

    # 実験者名を取得する
    git_name = "username: "
    git_name += subprocess.check_output(
        ["git", "config", "user.name"]
    ).decode().strip()

    # 上で取得したデータを log に記録する
    logger.info(git_info)
    logger.info(git_name)

    return logger


def get_os_info(logger: logging.Logger) -> logging.Logger:
    """
    OSの情報を取得し, 実験者環境を log ファイルに記録する関数

    Parameters
    ----------
    logger: logging.Logger
        実験の結果を記録する log データ

    Returns
    ----------
    logger: logging.Logger
        実験の結果を記録する log データ
    """
    # OS のスペックをまとめる
    os_info = "\n\n"
    spec = [
        f"\tOS: {platform.system()} {platform.release()}\n",
        f"\tProcessor: {platform.processor()}\n",
        f"\tMachine: {platform.machine()}\n",
        f"\tNode: {platform.node()}\n",
        f"\tPython Version: {platform.python_version()}\n",
    ]
    for item in spec:
        os_info += item

    # 上でまとめたスペックを log に記録する
    logger.info(f"OS infomation: {os_info}")

    return logger


def get_pip_list(logger: logging.Logger) -> logging.Logger:
    """
    使用ライブラリの情報を取得し, Pip リストを log ファイルに記録する関数

    Parameters
    ----------
    logger: logging.Logger
        実験の結果を記録する log データ

    Returns
    ----------
    logger: logging.Logger
        実験の結果を記録する log データ
    """
    # Pip リストを取得してまとめる
    pip_list = "pip3 list:\n\n"
    pip_list += subprocess.check_output(
        ["pip3", "list"]
    ).decode().strip()
    pip_list += "\n"

    # 上でまとめた Pip リストを log に記録する
    logger.info(pip_list)

    return logger


def start_experiment(cfg: dict) -> logging.Logger:
    """
    実験の環境をまとめて, log の記録を開始する関数

    Parameters
    ----------
    cfg: dict
        実験で参照する config データ

    Returns
    ----------
    logger: logging.Logger
        実験の結果を記録する log データ
    """
    # 実験開始時の時刻を取得し, 出力先のディレクトリを作成する
    date_time = datetime.datetime.now()
    cfg = get_directory(cfg, date_time)

    # log の初期化を行う
    logger = get_logger(cfg)

    # 各種機械学習フレームワークの乱数シードを固定する
    fix_seed(cfg["seed"])

    # 実験環境を log に記録する
    logger = get_git_info(logger)
    logger = get_os_info(logger)
    logger = get_pip_list(logger)

    return logger
