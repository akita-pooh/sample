import logging
from logging import FileHandler, Formatter, getLogger


def get_logger(cfg: dict) -> logging.Logger:
    """
    log の初期化を行う関数

    Parameters
    ----------
    cfg: dict
        実験で参照する config データ

    Returns
    ----------
    logger: logging.Logger
        実験の結果を記録する log データ
    """
    # logger インスタンスを作成し, log を初期化する
    logger = getLogger(__name__)
    logger.setLevel(logging.INFO)

    # config から handler と formatter を作成する
    handler = FileHandler(cfg["log"]["log_file"], mode="w")
    formatter = Formatter(cfg["log"]["log_formatter"])
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
