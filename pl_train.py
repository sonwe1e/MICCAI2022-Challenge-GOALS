import os.path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2.cv2
import torch
import argparse
from torch.utils.data import DataLoader
from utils import dice_loss, FocalLoss, SSIM_Loss, MSGMS_Loss
import model
from data import Dataset
import pytorch_lightning as pl
from segformer_pytorch import Segformer
import torchvision.transforms.functional as fn
from pytorch_lightning import loggers


class pointclassifier(pl.LightningModule):
    def __init__(self, loader_len):
        super().__init__()
        self.classifier = model.UNet()
        # self.classifier = Segformer(dims=(64, 128, 320, 512), ff_expansion=(4, 4, 4, 4), num_layers=(3, 6, 40, 3))
        self.lr = args.lr
        self.focal_loss = FocalLoss()
        self.entropy_loss = torch.nn.CrossEntropyLoss()
        self.ssim_loss = SSIM_Loss()
        self.gms_loss = MSGMS_Loss()
        self.step_size = loader_len * args.epochs / 2

    def forward(self, x):
        out = self.classifier(x)
        return out

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), weight_decay=5e-4, lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, max_lr=self.lr, base_lr=0.0,
        #                                               step_size_down=self.step_size, cycle_momentum=False)
        optimizer = torch.optim.SGD(self.parameters(), momentum=0.9, weight_decay=5e-4, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, max_lr=self.lr, base_lr=0.0,
                                                      step_size_down=self.step_size, cycle_momentum=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self.classifier(x)
        out = fn.resize(out, [800, 1024])
        loss1 = self.entropy_loss(out, y)
        loss2 = self.focal_loss(out, y)
        out = torch.argmax(out, dim=1)
        loss3 = dice_loss(out, y)
        IoU = torch.sum(out == y) / (args.batch_size * 1024 * 800)
        loss = loss1 + loss2 + loss3
        self.log('train.loss', loss)
        self.log('train.IoU', IoU)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        out = self.classifier(x)
        out = fn.resize(out, [800, 1104])
        loss1 = self.entropy_loss(out, y)
        loss2 = self.focal_loss(out, y)
        out = torch.argmax(out, dim=1)
        loss3 = dice_loss(out, y)
        loss = loss1 + loss2 + loss3
        IoU = torch.sum(out == y) / (args.batch_size * 1104 * 800)
        self.log('valid.loss', loss)
        self.log('valid.IoU', IoU)

    def predict_step(self, batch, batch_idx):
        x = batch
        out = self.classifier(x)
        out = torch.argmax(out, dim=1)
        out = out[0].cpu().detach().numpy()
        out[out == 1] = 80
        out[out == 2] = 160
        out[out == 3] = 255
        out = cv2.resize(out, (1100, 800), interpolation=cv2.INTER_NEAREST)
        if not os.path.exists('./Layer_Segmentations/'):
            os.makedirs('./Layer_Segmentations/')
        cv2.imwrite(f'./Layer_Segmentations/{batch_idx + 101:04d}.png', out)


parser = argparse.ArgumentParser(description='MICCAI')
parser.add_argument('--exp_name', type=str, default='UNet-PReLU-91-SGD-[1104,800]-SCAM', metavar='N',
                    help='Name of the experiment')
parser.add_argument('--batch_size', type=int, default=4, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--epochs', type=int, default=350, metavar='N',
                    help='number of episode to train ')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--device_ids', type=str, default='[1]',
                    help='induct fix id to train')
parser.add_argument('--test', type=bool, default=0,
                    help='decide whether to test')
parser.add_argument('--lr_find', type=bool, default=1,
                    help='decide whether to find lr')
args = parser.parse_args()


def main():
    pl.seed_everything(args.seed)
    wandb_logger = loggers.WandbLogger(name=args.exp_name, project='MICCAI')

    train_transform = A.Compose([
        A.Resize(height=800, width=1104, interpolation=cv2.INTER_CUBIC),
        A.ShiftScaleRotate(shift_limit=0.2, rotate_limit=30, scale_limit=0.2, p=0.5),
        A.Normalize(),
        ToTensorV2()
    ])
    test_transform = A.Compose([
        A.Resize(height=800, width=1104, interpolation=cv2.INTER_CUBIC),
        A.Normalize(),
        ToTensorV2()
    ])
    train_loader = DataLoader(Dataset(tvt='train', transform=train_transform), num_workers=16, pin_memory=True,
                              persistent_workers=True, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Dataset(tvt='val', transform=test_transform), num_workers=16, pin_memory=True,
                            persistent_workers=True, batch_size=args.batch_size, shuffle=False)

    model = pointclassifier(len(train_loader))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='valid.IoU', mode='max', save_top_k=3,
                                                       filename='{epoch}-{valid.loss:.3f}-{valid.IoU:.3f}',
                                                       dirpath='./MICCAI/' + args.exp_name + '/',
                                                       auto_insert_metric_name=False)

    if args.lr_find:
        trainer = pl.Trainer(gpus=eval(args.device_ids))
        lr_finder = trainer.tuner.lr_find(model, train_loader, val_loader, min_lr=1e-8, max_lr=1)
        print(lr_finder.suggestion())
        model.hparams.lr = lr_finder.suggestion()
    print(model.hparams)
    trainer = pl.Trainer(max_epochs=args.epochs,
                         devices=eval(args.device_ids),
                         accelerator='gpu',
                         callbacks=[checkpoint_callback],
                         fast_dev_run=args.test,
                         logger=wandb_logger,
                         log_every_n_steps=5,
                         # strategy='ddp',
                         )

    print('=========start training========')
    trainer.fit(model,
                train_loader,
                val_loader,
                )


if __name__ == '__main__':
    main()
