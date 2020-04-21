import argparse
import os

import json5
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from util.utils import initialize_config


def main(config, resume):
    torch.manual_seed(config["seed"])  # for both CPU and GPU
    np.random.seed(config["seed"])

    train_dataloader = DataLoader(
        dataset=initialize_config(config["train_dataset"]),
        batch_size=config["train_dataloader"]["batch_size"],
        num_workers=config["train_dataloader"]["num_workers"],
        shuffle=config["train_dataloader"]["shuffle"],
        pin_memory=config["train_dataloader"]["pin_memory"]
    )

    valid_dataloader = DataLoader(
        dataset=initialize_config(config["validation_dataset"]),
        num_workers=1,
        batch_size=1
    )

    model = initialize_config(config["model"])

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"])
    )

    loss_function = initialize_config(config["loss_function"])

    trainer_class = initialize_config(config["trainer"], pass_args=False)

    trainer = trainer_class(
        config=config,
        resume=resume,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=valid_dataloader
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-attention for Speech Enhancement")
    parser.add_argument(
        "-C", "--configuration",
        required=True,
        type=str,
        help="指定用于训练的配置文件 *.json5。"
    )
    parser.add_argument(
        "-O", "--omit_visualize_unprocessed_speech",
        action="store_true",
        help="每次实验开始时（首次或重启），会在验证集上计算基准性能（未处理时的）。 可以通过此选项跳过这个步骤。"
    )
    parser.add_argument(
        "-P", "--preloaded_model_path",
        type=str,
        help="预加载的模型路径。"
    )
    parser.add_argument(
        "-R", "--resume",
        action="store_true",
        help="Resume experiment from latest checkpoint."
    )
    args = parser.parse_args()

    if args.preloaded_model_path:
        assert args.resume == False, "Resume conflict with preloaded model. Please use one of them."

    configuration = json5.load(open(args.configuration))
    configuration["experiment_name"], _ = os.path.splitext(os.path.basename(args.configuration))
    configuration["config_path"] = args.configuration
    configuration["preloaded_model_path"] = args.preloaded_model_path
    configuration["omit_visualize_unprocessed_speech"] = args.omit_visualize_unprocessed_speech

    main(configuration, resume=args.resume)
