import lightgbm as lgb

from experiment_tools.set_up import start_experiment
from utils.cfg_diff import get_config, get_diff
from utils.preprocessing import make_datasets
from utils.result import extract_feature_importance, plot_data, predict


# config を取得
default_filename = "../config/default/LightGBM.json"
exp_filename = "../config/experiment/LightGBM.json"
_, cfg, default_str, exp_str = get_config(default_filename, exp_filename)

# log を起動し config の差分を取得
logger = start_experiment(cfg)
logger = get_diff(default_str, exp_str, logger)

# データセットの作成
out = make_datasets(cfg)
df = out["org_data"]
train_df = out["train_data"]
valid_df = out["valid_data"]
eval_df = out["eval_data"]
train_dataset = out["train_dataset"]
valid_dataset = out["valid_dataset"]

# データセットの記録
logger.info(f"original data records: {len(df)}")
logger.info(f"train data records: {len(train_df)}")
logger.info(f"valid data records: {len(valid_df)}")
logger.info(f"eval data records: {len(eval_df)}")
logger.info(f"train raw data:\n\n{train_df}\n")

# 評価指標のログを保存する辞書を用意
training_data = {}

# モデルの学習
model = lgb.train(
    cfg["params"],
    train_dataset,
    valid_sets=[train_dataset, valid_dataset],
    valid_names=["Train", "Valid"],
    num_boost_round=cfg["training"]["num_boost_round"],
    callbacks=[
        lgb.early_stopping(
            stopping_rounds=cfg["training"]["early_stopping"]["stopping_rounds"],
            verbose=cfg["training"]["early_stopping"]["verbose"]
        ),
        lgb.log_evaluation(cfg["training"]["verbose_eval"]),
        lgb.record_evaluation(training_data)
    ]
)

# 結果の出力先ディレクトリを指定
out_dir = cfg["log"]["log_file"].replace("LightGBM.log", "")

# 学習過程のロスを描画
plot_data(cfg, training_data, out_dir)

# 特徴量の重要度を描画
extract_feature_importance(model, out_dir)

# 予測結果を描画
predict(eval_df, model, out_dir)
