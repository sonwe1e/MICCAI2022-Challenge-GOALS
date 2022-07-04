
from data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import cv2
from pl_train import pointclassifier
import albumentations as A
from albumentations.pytorch import ToTensorV2

model = pointclassifier(1)
test_transform = A.Compose([
    A.Resize(height=768, width=1024, interpolation=cv2.INTER_CUBIC),
    A.Normalize(),
    ToTensorV2()
])
test_loader = DataLoader(Dataset(tvt='test', transform=test_transform), num_workers=16, pin_memory=True,
                         persistent_workers=True, batch_size=1, shuffle=False)
trainer = pl.Trainer()
trainer.predict(model, test_loader,
                ckpt_path='/home/gdut403/sonwe1e/MICCAI/MICCAI/UNet-PReLU-91-SGD/278-0.016-0.994.ckpt',)
