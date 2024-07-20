import numpy as np
import torch
import torch.utils.data
import torchvision.transforms.functional
from torch import nn

from labml import lab, tracker, experiment, monit
from labml.configs import BaseConfigs
from labml_helpers.device import DeviceConfigs
from labml_nn.unet.carvana import CarvanaDataset
from labml_nn.unet import UNet


class Configs(BaseConfigs):

    device: torch.device = DeviceConfigs()
    model: UNet
    image_channels: int = 3
    mask_channels: int = 1
    batch_size: int = 1
    learning_rate: float = 2.5e-4
    epochs: int = 4
    dataset: CarvanaDataset
    data_loader: torch.utils.data.DataLoader
    loss_func = nn.BCELoss()
    sigmoid = nn.Sigmoid()
    optimizer: torch.optim.Adam

    def init(self):
        self.dataset = CarvanaDataset(lab.get_data_path() / 'carvana' / 'train',
                                      lab.get_data_path() / 'carvana' / 'train_masks')
        self.model = UNet(self.image_channels, self.mask_channels).to(self.device)
        self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size,
                                                       shuffle=True, pin_memory=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        tracker.set_image('sample', True)

    @torch.no_grad()
    def sample(self, idx=-1):
        x, _ = self.dataset[np.random.randint(len(self.dataset))]
        x = x.to(self.device)
        mask = self.sigmoid(self.model(x[None, :]))
        x = torchvision.transforms.functional.center_crop(x, [mask.shape[2], mask.shape[3]])
        tracker.save('sample', x * mask)

    def train(self):
        for _, (image, mask) in monit.mix(('Train', self.data_loader), (self.sample, list(range(50)))):
            tracker.add_global_step()
            image, mask = image.to(self.device), mask.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(image)
            mask = torchvision.transforms.functional.center_crop(mask, [logits.shape[2], logits.shape[3]])
            loss = self.loss_func(self.sigmoid(logits), mask)
            loss.backward()
            self.optimizer.step()
            tracker.save('loss', loss)

    def run(self):
        for _ in monit.loop(self.epochs):
            self.train()
            tracker.new_line()
            experiment.save_checkpoint()


def main():
    experiment.create(name='unet')
    configs = Configs()
    experiment.configs(configs, {})
    configs.init()
    experiment.add_pytorch_models({'model': configs.model})
    with experiment.start():
        configs.run()


if __name__ == '__main__':
    main()