import datetime
import os
import random
import traceback

import hydra
import pyrootutils
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image
import torchvision

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.dataset import DataloaderMode, create_dataloader
from src.model import Model, create_model
from src.tools.test_model import test_model
from src.utils.loss import get_loss
from src.utils.utils import get_logger, is_logging_process, set_random_seed
from src.utils.writer import Writer


def setup(cfg, rank):
    os.environ["MASTER_ADDR"] = cfg.dist.master_addr
    os.environ["MASTER_PORT"] = cfg.dist.master_port
    timeout_sec = 1800
    if cfg.dist.timeout is not None:
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        timeout_sec = cfg.dist.timeout
    timeout = datetime.timedelta(seconds=timeout_sec)

    # initialize the process group
    dist.init_process_group(
        cfg.dist.mode,
        rank=rank,
        world_size=cfg.dist.gpus,
        timeout=timeout,
    )


def img_process(image_path):
    img = Image.open(image_path).convert("RGB")
    transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomCrop(size=224),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    patches_1 = transforms(img)
    patches_2 = transforms(img)
    batch_patches = torch.Tensor(2, 3, 224, 224)
    batch_patches[0,:,:,:] = patches_1.unsqueeze(0)
    batch_patches[1,:,:,:] = patches_2.unsqueeze(0)
    return batch_patches


def cleanup():
    dist.destroy_process_group()


def distributed_run(fn, cfg):
    mp.spawn(fn, args=(cfg,), nprocs=cfg.dist.gpus, join=True)


def inference_single(rank,cfg, image_path):#rank=0
    if cfg.dist.device == "cuda" and cfg.dist.gpus != 0:
        cfg.dist.device = rank
        setup(cfg, rank)
        torch.cuda.set_device(cfg.dist.device)

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    # init Model
    net_arch = create_model(cfg=cfg)
    loss_f = get_loss(cfg=cfg)
    model = Model(cfg, net_arch, loss_f, rank)

    # process the image
    batch_patches = img_process(image_path)

    # load training state / network checkpoint
    assert cfg.load.network_chkpt_path is not None
    print("hhuhuh")
    model.load_network()
    scores = model.inference(batch_patches)
    print("Score: ", scores)
    print("DONE")

    if cfg.dist.device == "cuda" and cfg.dist.gpus != 0:
        cleanup()


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(hydra_cfg: DictConfig):
    hydra_cfg.dist.device = hydra_cfg.dist.device.lower()
    with open_dict(hydra_cfg):
        hydra_cfg.job_logging_cfg = HydraConfig.get().job_logging
        hydra_cfg.hydra_output_dir = HydraConfig.get().run.dir
    # random seed
    if hydra_cfg.random_seed is None:
        hydra_cfg.random_seed = random.randint(1, 10000)
    set_random_seed(hydra_cfg.random_seed)
    
    image_path = 'src/I81_01_05.png'

    if hydra_cfg.dist.device == "cuda" and hydra_cfg.dist.gpus < 0:
        hydra_cfg.dist.gpus = torch.cuda.device_count()
    if hydra_cfg.dist.device == "cpu" or hydra_cfg.dist.gpus == 0:
        hydra_cfg.dist.gpus = 0
        #print(hydra_cfg)
        inference_single(0, hydra_cfg, image_path)
    else:
        distributed_run(test_loop, hydra_cfg)


if __name__ == "__main__":
    main()
