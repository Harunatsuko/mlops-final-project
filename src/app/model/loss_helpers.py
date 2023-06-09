import torch

def disc_loss(real_A, fake_A, disc_A, adv_criterion, device):
    pred_real = disc_A(real_A).to(device)
    pred_fake = disc_A(fake_A.detach()).to(device)
    real_label = torch.ones(pred_real.size()).to(device)
    fake_label = torch.zeros(pred_fake.size()).to(device)
    loss_real = adv_criterion(pred_real, real_label)
    loss_fake = adv_criterion(pred_fake, fake_label)
    disc_loss = (loss_real + loss_fake) /2
    return disc_loss

def gen_adversarial_loss(real_A, disc_B, gen_AB, adv_criterion):
    fake_B = gen_AB(real_A)
    pred_fake = disc_B(fake_B)
    adversarial_loss = adv_criterion(pred_fake, torch.ones_like(pred_fake))
    return adversarial_loss, fake_B

def identity_loss(real_A, gen_BA, identity_criterion):
    identity_A = gen_BA(real_A)
    identity_loss = identity_criterion(identity_A, real_A)
    return identity_loss, identity_A

def cycle_consistency_loss(real_A, fake_B, gen_BA, cycle_criterion):
    cycle_A = gen_BA(fake_B)
    cycle_loss = cycle_criterion(cycle_A, real_A)
    return cycle_loss, cycle_A

def gen_loss(real_A,
                 real_B,
                 gen_AB,
                 gen_BA,
                 disc_A,
                 disc_B,
                 adv_criterion,
                 identity_criterion,
                 cycle_criterion,
                 lambda_identity=0.1,
                 lambda_cycle=10):
    adv_loss_a, fake_A = gen_adversarial_loss(real_B, disc_A, gen_BA, adv_criterion)
    adv_loss_b, fake_B = gen_adversarial_loss(real_A, disc_B, gen_AB, adv_criterion)
    adv_loss = adv_loss_a + adv_loss_b 

    idn_loss_a, idn_a = identity_loss(real_A, gen_BA, identity_criterion)
    idn_loss_b, idn_b = identity_loss(real_B, gen_AB, identity_criterion)
    idn_loss = idn_loss_a + idn_loss_b

    cycle_loss_a, cycle_a = cycle_consistency_loss(real_A, fake_B, gen_BA, cycle_criterion)
    cycle_loss_b, cycle_b = cycle_consistency_loss(real_B, fake_A, gen_AB, cycle_criterion)
    cycle_loss = cycle_loss_a + cycle_loss_b
    
    gen_loss = lambda_identity * idn_loss + lambda_cycle * cycle_loss + adv_loss

    return gen_loss, fake_A, fake_B