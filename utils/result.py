import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd


def plot_data(
        cfg: dict, training_data: dict, out_dir: str
    ) -> None:
    """
    学習過程のロスの遷移を表示する関数

    Parameters
    ----------
    cfg: dict
        実験に使う値の config データ
    training_data: dict
        学習過程のロスのデータ
    out_dir: str
        結果を保存するディレクトリのパス

    Returns
    ----------
    None
    """
    # 学習で得られたデータをまとめる
    metric = cfg["params"]["metric"]
    train_y = [
        item for item in training_data["Train"][metric]
    ]
    valid_y = [
        item for item in training_data["Valid"][metric]
    ]
    x = [i + 1 for i in range(len(train_y))]

    # 画像のスタイルを指定する
    plt.figure(figsize=(18, 12))
    plt.title("Loss comparison", size=15, color="red")
    plt.grid()

    # データのプロットをする
    plt.plot(x, train_y, label="Train")
    plt.plot(x, valid_y, label="Valid")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # 画像を保存する
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.)
    plt.savefig(f"{out_dir}/loss_curve.png")


def extract_feature_importance(
        model: lgb.basic.Booster, out_dir: str
    ) -> None:
    """
    特徴量の重要度を表示する関数

    Parameters
    ----------
    model: lgb.basic.Booster
        実験で学習させたモデルのインスタンス
    out_dir: str
        結果を保存するディレクトリのパス

    Returns
    ----------
    None
    """
    # 画像のスタイルを指定する
    plt.figure(figsize=(18, 12))
    plt.title("LightGBM Feature Importance", size=15, color="red")
    plt.grid()

    # 特徴量の重要度を描画する
    lgb.plot_importance(model)

    # 画像を保存する
    plt.savefig(f"{out_dir}/feature_importance.png")


def predict(
        eval_df: pd.DataFrame, model: lgb.basic.Booster, out_dir: str
    ) -> None:
    """
    データの予測をする関数

    Parameters
    ----------
    eval_df: pd.DataFrame
        評価用のデータフレーム
    model: lgb.basic.Booster
        実験で学習させたモデルのインスタンス
    out_dir: str
        結果を保存するディレクトリのパス

    Returns
    ----------
    None
    """
    # データの準備をする
    x = eval_df["x"].to_list()
    y_true = eval_df["y"]
    y_data = list(eval_df.drop(["x", "y"], axis=1).iloc[0])

    # 再帰的に予測する
    y_preds = []
    for i in range(len(eval_df)):
        tmp_dataset = [x[i]] + y_data
        y_pred = model.predict(
            [tmp_dataset], num_iteration=model.best_iteration
        )[0]
        y_preds.append(y_pred)
        y_data = [y_pred] + y_data[:-1]
    
    # 画像のスタイルを指定する
    plt.figure(figsize=(18, 12))
    plt.title("Prediction", size=15, color="red")
    plt.grid()

    # データのプロットをする
    plt.scatter(x, y_true, label="True")
    plt.plot(x, y_preds, label="Preds", color="orange")

    # 画像を保存する
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.)
    plt.savefig(f"{out_dir}/prediction.png")
