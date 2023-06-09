import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchvision.utils import make_grid
from model.generator import Generator
from model.discriminator import Discriminator
from model.loss_helpers import disc_loss, gen_loss

MODEL_FILENAME = 'cycleGANmodel.pth'

class CycleGANModel():
    def __init__(self,
                 device,
                 output_dir,
                 dim_A = 3, 
                 dim_B = 3,
                 pretrained_path = None,
                 load_shape = 260,
                 target_shape = 256):
        self.device = device
        self.output_dir = output_dir
        self.load_shape = load_shape
        self.target_shape = target_shape
        self.gen_AB = Generator(dim_A, dim_B).to(device)
        self.gen_BA = Generator(dim_B, dim_A).to(device)
        self.dim_A = dim_A
        self.dim_B = dim_B
        
        self.disc_A = Discriminator(dim_A).to(device)
        
        self.disc_B = Discriminator(dim_B).to(device)
        

        def _weights_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
                torch.nn.init.constant_(m.bias, 0)

        if pretrained_path is not None:
            self.pre_dict = torch.load(pretrained_path)
            self.gen_AB.load_state_dict(pre_dict['gen_AB'])
            self.gen_BA.load_state_dict(pre_dict['gen_BA'])
            self.gen_opt.load_state_dict(pre_dict['gen_opt'])
            self.disc_A.load_state_dict(pre_dict['disc_A'])
            self.disc_A_opt.load_state_dict(pre_dict['disc_A_opt'])
            self.disc_B.load_state_dict(pre_dict['disc_B'])
            self.disc_B_opt.load_state_dict(pre_dict['disc_B_opt'])
        else:
            self.gen_AB = self.gen_AB.apply(_weights_init)
            self.gen_BA = self.gen_BA.apply(_weights_init)
            self.disc_A = self.disc_A.apply(_weights_init)
            self.disc_B = self.disc_B.apply(_weights_init)


    def _train_step(self, real_A, real_B, adv_criterion, recon_criterion):
        real_A = nn.functional.interpolate(real_A, size=self.target_shape)
        real_B = nn.functional.interpolate(real_B, size=self.target_shape)
        cur_batch_size = len(real_A)
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)

        # discriminators
        self.disc_A_opt.zero_grad() 
        with torch.no_grad():
            fake_A = self.gen_BA(real_B)
        
        disc_A_loss = disc_loss(real_A, fake_A, self.disc_A, adv_criterion, self.device)
        disc_A_loss.backward(retain_graph=True)
        self.disc_A_opt.step()

        self.disc_B_opt.zero_grad()
        with torch.no_grad():
            fake_B = self.gen_AB(real_A)
        disc_B_loss = disc_loss(real_B, fake_B, self.disc_B, adv_criterion, self.device)
        disc_B_loss.backward(retain_graph=True)
        self.disc_B_opt.step()

        # generator
        self.gen_opt.zero_grad()
        gen_loss_val, fake_A, fake_B = gen_loss(
            real_A, real_B, self.gen_AB, self.gen_BA, self.disc_A, self.disc_B,
             adv_criterion, recon_criterion, recon_criterion
        )
        gen_loss_val.backward() 
        self.gen_opt.step()

        return disc_A_loss, disc_B_loss, gen_loss_val, fake_A, fake_B

    def train(self, dataset, num_epochs, lr, batch_size, verbose=False, display_step=100):
        self.gen_opt = torch.optim.Adam(list(self.gen_AB.parameters()) + list(self.gen_BA.parameters()), lr=lr, betas=(0.5, 0.999))
        self.disc_A_opt = torch.optim.Adam(self.disc_A.parameters(), lr=lr, betas=(0.5, 0.999))
        self.disc_B_opt = torch.optim.Adam(self.disc_B.parameters(), lr=lr, betas=(0.5, 0.999))
        
        mean_gen_loss = 0
        mean_disc_loss = 0
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            cur_step = 0
            for real_A, real_B in tqdm(dataloader):
                
                disc_A_loss, disc_B_loss, gen_loss, fa, fb = self._train_step(real_A,
                                                                      real_B,
                                                                      nn.MSELoss(),
                                                                      nn.L1Loss())

                if verbose:
                    mean_disc_loss += disc_A_loss.item() / display_step
                    mean_gen_loss += gen_loss.item() / display_step

                    if cur_step % display_step == 0:
                        print(f"epoch {epoch}: step {cur_step}")
                        print(f"gen_loss: {mean_gen_loss}, disc_loss: {mean_disc_loss}")
                    
                        mean_generator_loss = 0
                        mean_discriminator_loss = 0

                
                cur_step += 1
            self.save()

    def save(self):
        path_to_save = self.output_dir + '/' + MODEL_FILENAME
        torch.save({
            'gen_AB': self.gen_AB.state_dict(),
            'gen_BA': self.gen_BA.state_dict(),
            'gen_opt': self.gen_opt.state_dict(),
            'disc_A': self.disc_A.state_dict(),
            'disc_A_opt': self.disc_A_opt.state_dict(),
            'disc_B': self.disc_B.state_dict(),
            'disc_B_opt': self.disc_B_opt.state_dict(),
        }, path_to_save)

    def infer(self, real, a2b = True):
        if a2b:
            with torch.no_grad():
                image_tensor = gen_AB(real)
        else:
            with torch.no_grad():
                image_tensor = gen_BA(real)
        image_tensor = (image_tensor + 1) / 2
        image_unflat = image_tensor.detach().cpu().view(-1, *size)
        image = image_unflat.permute(1, 2, 0).squeeze()
        return image
