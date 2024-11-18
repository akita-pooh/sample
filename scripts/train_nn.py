import torch
from torch.utils.data import DataLoader

from experiment_tools.set_up import start_experiment
from models.networks import NeuralNetwork
from trainers.loop import train_nn
from trainers.opt import options
from utils.cfg_diff import get_config, get_diff
from utils.preprocessing import make_datasets_for_nn
from utils.result import plot_data, predict


# config を取得
default_filename = "../config/default/NeuralNetwork.json"
exp_filename = "../config/experiment/NeuralNetwork.json"
_, cfg, default_str, exp_str = get_config(default_filename, exp_filename)

# log を起動し config の差分を取得
logger = start_experiment(cfg)
logger = get_diff(default_str, exp_str, logger)

# プロセッサーの指定
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"device: {device}")

# データセットの作成
out = make_datasets_for_nn(cfg)
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

# 各 DataLoader の作成
train_dataloader = DataLoader(
    dataset=out["train_dataset"],
    batch_size=cfg["dataloader_params"]["batch_size"],
    shuffle=cfg["dataloader_params"]["shuffle"],
    sampler=cfg["dataloader_params"]["sampler"],
    batch_sampler=cfg["dataloader_params"]["batch_sampler"],
    num_workers=cfg["dataloader_params"]["num_workers"],
    collate_fn=cfg["dataloader_params"]["collate_fn"],
    pin_memory=cfg["dataloader_params"]["pin_memory"],
    drop_last=cfg["dataloader_params"]["drop_last"],
    timeout=cfg["dataloader_params"]["timeout"],
    worker_init_fn=cfg["dataloader_params"]["worker_init_fn"],
    prefetch_factor=cfg["dataloader_params"]["prefetch_factor"],
    persistent_workers=cfg["dataloader_params"]["persistent_workers"],
    pin_memory_device=cfg["dataloader_params"]["pin_memory_device"]
)
valid_dataloader = DataLoader(
    dataset=out["valid_dataset"],
    batch_size=cfg["dataloader_params"]["batch_size"],
    shuffle=cfg["dataloader_params"]["shuffle"],
    sampler=cfg["dataloader_params"]["sampler"],
    batch_sampler=cfg["dataloader_params"]["batch_sampler"],
    num_workers=cfg["dataloader_params"]["num_workers"],
    collate_fn=cfg["dataloader_params"]["collate_fn"],
    pin_memory=cfg["dataloader_params"]["pin_memory"],
    drop_last=cfg["dataloader_params"]["drop_last"],
    timeout=cfg["dataloader_params"]["timeout"],
    worker_init_fn=cfg["dataloader_params"]["worker_init_fn"],
    prefetch_factor=cfg["dataloader_params"]["prefetch_factor"],
    persistent_workers=cfg["dataloader_params"]["persistent_workers"],
    pin_memory_device=cfg["dataloader_params"]["pin_memory_device"]
)

# モデルの構築
input_dim = cfg["lag"] + 1
output_dim = 1
model = NeuralNetwork(cfg, input_dim, output_dim).to(device=device)
logger.info(f"model architecture:\n\n{model}\n")

# 最適化器の設定
opt = options(cfg, model)
optimizer = opt.getter()

# エポック数の取得
epochs = cfg["params"]["epochs"]

# モデルの学習
model, training_data = train_nn(
    epochs=epochs,
    train_dataloader=train_dataloader,
    valid_dataloader=valid_dataloader,
    model=model,
    optimizer=optimizer,
    batch_size=cfg["dataloader_params"]["batch_size"],
    device=device,
    cfg=cfg
)

# 結果の出力先ディレクトリを指定
out_dir = cfg["log"]["log_file"].replace("NeuralNetwork.log", "")

# 学習過程のロスを描画
plot_data(cfg, training_data, out_dir, "NeuralNetwork")

# モデルの保存
torch.save(model.state_dict(), f"{out_dir}model_weight.pth")

# 予測結果を描画
predict(eval_df, "NeuralNetwork", out_dir, model)
