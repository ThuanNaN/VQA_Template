import os
import copy
import time
import logging
import argparse
import yaml
from yaml.loader import SafeLoader
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
from utils import seed_everything, colorstr
from torch.optim.lr_scheduler import MultiStepLR
from dataset import ViVQADataset, OpenViVQADataset
from transformers import AutoTokenizer, AutoProcessor
from torch.utils.data import DataLoader
from models import SimpleVQAConfig, SimpleVQA

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(message)s", level=logging.INFO)
LOGGER = logging.getLogger("VQA_AUG")

def train_model(model, dataloaders, optimizer, opt, wandb, lr_scheduler=None):
    since = time.perf_counter()
    num_epochs, device = opt.epochs, opt.device
    LOGGER.info(f"\n{colorstr('Hyperparameter:')} {opt}")
    LOGGER.info(f"\n{colorstr('Device:')} {device}")
    LOGGER.info(f"\n{colorstr('Optimizer:')} {optimizer}")

    if opt.lr_scheduler:
        LOGGER.info(
            f"\n{colorstr('LR Scheduler:')} {type(lr_scheduler).__name__}")
    else:
        lr_scheduler = None
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=list(
            range(torch.cuda.device_count())))

    criterion = nn.CrossEntropyLoss()
    LOGGER.info(f"\n{colorstr('Loss:')} {type(criterion).__name__}")

    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "lr": []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_model_optim = copy.deepcopy(optimizer.state_dict())
    best_val_acc = 0.0

    model.to(device)
    for epoch in range(num_epochs):
        LOGGER.info(colorstr(f'\nEpoch {epoch}/{num_epochs-1}:'))
        for phase in ["train", "val"]:
            if phase == "train":
                LOGGER.info(colorstr('bright_yellow', 'bold', '\n%20s' + '%15s' * 3) %
                            ('Training:', 'gpu_mem', 'loss', 'acc'))
                model.train()
            else:
                LOGGER.info(colorstr('bright_yellow', 'bold', '\n%20s' + '%15s' * 3) %
                            ('Validation:', 'gpu_mem', 'loss', 'acc'))
                model.eval()
            running_items = 0
            running_loss = 0.0
            running_corrects = 0
            _phase = tqdm(dataloaders[phase],
                          total=len(dataloaders[phase]),
                          bar_format='{desc} {percentage:>7.0f}%|{bar:10}{r_bar}{bar:-10b}',
                          unit='batch')

            for inputs, labels in _phase:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        history['lr'].append(lr_scheduler.optimizer.param_groups[0]
                                             ["lr"]) if lr_scheduler else history['lr'].append(opt.lr)
                        if lr_scheduler is not None:
                            lr_scheduler.step()
                running_items += inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                epoch_loss = running_loss / running_items
                epoch_acc = running_corrects / running_items
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}GB'
                desc = ('%35s' + '%15.6g' * 2) % (mem, running_loss /
                                                  running_items, running_corrects / running_items)
                _phase.set_description_str(desc)

            if phase == 'train':
                if opt.wandb_log:
                    wandb.log({"train_acc": epoch_acc, "train_loss": epoch_loss}, step = epoch)
                    if lr_scheduler:
                        wandb.log(
                            {"lr": lr_scheduler.optimizer.param_groups[0]["lr"]}, step = epoch)
                    else:
                        wandb.log({"lr": opt.lr})
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc.item())
            else:
                if opt.wandb_log:
                    wandb.log({"val_acc": epoch_acc, "val_loss": epoch_loss}, step = epoch)
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc.item())
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_model_optim = copy.deepcopy(optimizer.state_dict())

    time_elapsed = time.perf_counter() - since
    LOGGER.info(f"Training complete in \
                {time_elapsed // 3600}h \
                {time_elapsed % 3600 // 60}m \
                { time_elapsed % 60}s with \
                {num_epochs} epochs")
    LOGGER.info(f"Best val Acc: {round(best_val_acc.item(), 6)}")
    model.load_state_dict(best_model_wts)
    optimizer.load_state_dict(best_model_optim)

    return model, best_val_acc.item()


def test_model(model, test_loader, device):
    model.to(device)
    model.eval()
    totals = 0
    corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            totals += inputs.size(0)
            corrects += torch.sum(preds == labels.data)

    acc = corrects / totals
    return acc.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='cuda device or cpu (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed will start at seed = 2 (default: %(default)s)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Mini-batch size for each iteration when training model (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train model (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate (default: %(default)s)')
    parser.add_argument('--wandb_log', action='store_true',
                        help='Log training process to wandb')
    parser.add_argument('--wandb_name', type=str, default='VQA-Tempate',
                        help='Name of wandb project (default: %(default)s)')
    parser.add_argument('--name', type=str, default='exp',
                        help='Name of the run (default: %(default)s)')
    opt = parser.parse_args()
    seed_everything(opt.seed)

    try:
        device_name = os.getlogin()
    except:
        device_name = "Colab/Cloud"
    if opt.wandb_log:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            project=opt.wandb_name,
            name=opt.name,
            tags=[device_name],
            config=vars(opt))
    else:
        wandb = None

    text_processor = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    img_processor = AutoProcessor.from_pretrained('google/vit-base-patch16-224')

    train_dataset = ViVQADataset(
        ann_path="data/vivqa/train.csv",
        img_dir="data/vivqa/images",
        text_processor=text_processor,
        img_processor=img_processor,
    )
    val_dataset = ViVQADataset(
        ann_path="data/vivqa/test.csv",
        img_dir="data/vivqa/images",
        text_processor=text_processor,
        img_processor=img_processor,
    )
    num_classes = len(train_dataset.get_label_encoder())
    LOGGER.info(f"Number of classes: {num_classes}")

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    dataloaders = {
        "train": train_loader,
        "val": val_loader,
    }
    
    config = SimpleVQAConfig(
        vis_model_name='google/vit-base-patch16-224',
        text_model_name='vinai/phobert-base-v2',
        hidden_size=768,
        num_classes=num_classes
    )
    model = SimpleVQA(config)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=opt.lr,
                                  weight_decay=opt.weight_decay)

    best_model, val_acc = train_model(model=model,
                                      dataloaders=dataloaders,
                                      optimizer=optimizer,
                                      opt=opt,
                                      wandb=wandb,
                                      lr_scheduler=None)
    test_acc = test_model(best_model, dataloaders["test"], opt.device)
    LOGGER.info(f"Validation accuracy: {round(val_acc, 6)}")
    LOGGER.info(f"Test accuracy: {round(test_acc, 6)}")
    if opt.wandb_log:
        wandb.finish()
