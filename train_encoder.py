import os
import wandb
import getpass
import random, torch, numpy

import brownian_system, brownian_bridge_system

import pytorch_lightning as pl
import recovery
import hydra
from pytorch_lightning.strategies import DDPStrategy

torch.backends.cudnn.benchmark = True

SYSTEM = {
    'brownian_bridge': brownian_bridge_system.BrownianBridgeSystem,
    'brownian': brownian_system.BrownianSystem
    # 'vae': vae_system.VAESystem,
    # 'infonce': infonce_system.InfoNCESystem
}

@hydra.main(config_path="./config/encoder", config_name="brownian_bridge")
def run(config):
    if config.wandb_settings.dryrun:
        print("Running in dryrun mode")
        os.environ['WANDB_MODE'] = 'dryrun'
    os.environ['WANDB_CONSOLE']='wrap'

    seed_everything(
        config.experiment_params.seed,
        use_cuda=config.experiment_params.cuda)

    wandb.init(
        project=config.wandb_settings.project,
        entity=getpass.getuser(),
        name=config.wandb_settings.exp_name,
        group=config.wandb_settings.group,
        config=config,
    )

    print("CKPT AT {}".format(
        os.path.join(config.wandb_settings.exp_name,
        'checkpoints')))
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(config.wandb_settings.exp_name, 'checkpoints'),
        save_top_k=-1,
        every_n_epochs=config.experiment_params.checkpoint_epochs,
    )

    SystemClass = SYSTEM[config.loss_params.loss]
    system = SystemClass(config)

    trainer = pl.Trainer(
        default_root_dir=config.wandb_settings.exp_dir,
        devices=1,
        num_nodes=config.experiment_params.num_nodes,
        # strategy=DDPStrategy(find_unused_parameters=True),
        accelerator="gpu",
        accumulate_grad_batches = 8, # 累加多个batch的梯度
        # precision = 16,     #半精度
        auto_scale_batch_size = 'power', #自动搜索最大batch_size
        auto_lr_find=True,  # 最优学习率发现
        callbacks=ckpt_callback,
        max_epochs=int(config.experiment_params.num_epochs),
        min_epochs=int(config.experiment_params.num_epochs),
    )

    trainer.fit(system)

    ## Save the model
    system.save(directory=wandb.run.dir)

    ## Evaluation:
    trainer.test(system)

def seed_everything(seed, use_cuda=True):
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda: torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":
    run()



