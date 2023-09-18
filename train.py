from model_def import Discriminator, ECNet
from loss import RecLoss, PyrLoss, AdvLoss
from dataLoader import make_loader
import json
import torch as t
import os
from torch import nn, optim
from tools import calc_pnsr


def train_epoch(model, discriminator, recLoss, pyrLoss, advLoss, d_criterion, train_loader, current_epoch, begin_use_adv_loss_epoch, g_optimizer, d_optimizer, device_ids, epochs):
    steps = len(train_loader)
    current_step = 1
    for d_laplacian, l_gaussian in train_loader:
        model.train()
        discriminator.eval()
        d_laplacian = [i.cuda(device_ids[0]) for i in d_laplacian]
        l_gaussian = [i.cuda(device_ids[0]) for i in l_gaussian]
        model_outputs = model(d_laplacian)
        rec_loss = recLoss(model_outputs, l_gaussian)
        pyr_loss = pyrLoss(model_outputs, l_gaussian)
        if begin_use_adv_loss_epoch <= current_epoch:
            adv_loss = advLoss(model_outputs, discriminator)
        else:
            adv_loss = t.tensor(0).type(t.FloatTensor).to(pyr_loss.device)
        g_total_loss = rec_loss + pyr_loss + adv_loss
        g_optimizer.zero_grad()
        g_total_loss.backward()
        g_optimizer.step()
        pnsr = calc_pnsr(model_outputs, l_gaussian)
        if begin_use_adv_loss_epoch <= current_epoch:
            model.eval()
            discriminator.train()
            with t.no_grad():
                model_outputs = model(d_laplacian)
            discriminator_input = t.cat([model_outputs[-1], l_gaussian[-1]], dim=0)
            discriminator_label = t.cat([t.tensor([0] * model_outputs[-1].size()[0]), t.tensor([1] * model_outputs[-1].size()[0])], dim=0).type(t.LongTensor).to(discriminator_input.device)
            discriminator_out = discriminator(discriminator_input)
            d_total_loss = d_criterion(discriminator_out, discriminator_label)
            d_optimizer.zero_grad()
            d_total_loss.backward()
            d_optimizer.step()
            if current_step % 5 == 0:
                print(
                    "epoch:%d/%d, step:%d/%d, rec_loss:%.5f, pyr_loss:%.5f, adv_loss:%.5f, d_loss:%.5f, g_total_loss:%.5f, psnr:%.5f" % (
                    current_epoch, epochs, current_step, steps, rec_loss.item(), pyr_loss.item(), adv_loss.item(), d_total_loss.item(),
                    g_total_loss.item(), pnsr.item()))
        else:
            if current_step % 5 == 0:
                print(
                    "epoch:%d/%d, step:%d/%d, rec_loss:%.5f, pyr_loss:%.5f, adv_loss:%.5f, d_loss:%.5f, g_total_loss:%.5f, psnr:%.5f" % (
                    current_epoch, epochs, current_step, steps, rec_loss.item(), pyr_loss.item(), adv_loss.item(), 0,
                    g_total_loss.item(), pnsr.item()))
        current_step += 1
    print("saving epoch model......")
    t.save(model.module.state_dict(), os.path.join(model_save_dir, "epoch.pth"))
    return model, discriminator


def valid_epoch(model, discriminator, recLoss, pyrLoss, advLoss, valid_loader, current_eopch, begin_use_adv_loss_epoch, device_ids):
    global best_psnr
    model.eval()
    discriminator.eval()
    steps = len(valid_loader)
    accum_recloss = 0
    accum_pyrloss = 0
    accum_advloss = 0
    accum_psnr = 0
    for d_laplacian, l_gaussian in valid_loader:
        d_laplacian = [i.cuda(device_ids[0]) for i in d_laplacian]
        l_gaussian = [i.cuda(device_ids[0]) for i in l_gaussian]
        with t.no_grad():
            model_outputs = model(d_laplacian)
            psnr = calc_pnsr(model_outputs, l_gaussian)
            rec_loss = recLoss(model_outputs, l_gaussian)
            pyr_loss = pyrLoss(model_outputs, l_gaussian)
            if current_eopch >= begin_use_adv_loss_epoch:
                adv_loss = advLoss(model_outputs, discriminator)
            else:
                adv_loss = t.tensor(0).type(t.FloatTensor).to(pyr_loss.device)
            accum_recloss += rec_loss.item()
            accum_pyrloss += pyr_loss.item()
            accum_advloss += adv_loss.item()
            accum_psnr += psnr.item()
    avg_rec_loss = accum_recloss / steps
    avg_pyr_loss = accum_pyrloss / steps
    avg_adv_loss = accum_advloss / steps
    avg_psnr = accum_psnr / steps
    avg_total_loss = avg_rec_loss + avg_pyr_loss + avg_adv_loss
    print("###########valid epoch:%d###########" % (current_eopch,))
    print("rec_loss:%.5f, pyr_loss:%.5f, adv_loss:%.5f, g_total_loss:%.5f, psnr:%.5f" % (avg_rec_loss, avg_pyr_loss, avg_adv_loss, avg_total_loss, avg_psnr))
    if avg_psnr > best_psnr:
        print("saving best model......")
        best_psnr = avg_psnr
        t.save(model.module.state_dict(), os.path.join(model_save_dir, "best.pth"))
    return model, discriminator


def main():
    model = ECNet(laplacian_level_count, layer_count_of_every_unet, first_layer_out_channels_of_every_unet)
    discriminator = Discriminator(discriminator_image_size)
    model = nn.DataParallel(module=model, device_ids=device_ids)
    discriminator = nn.DataParallel(module=discriminator, device_ids=device_ids)
    model = model.cuda(device_ids[0])
    discriminator = discriminator.cuda(device_ids[0])
    recLoss = RecLoss().cuda(device_ids[0])
    pyrLoss = PyrLoss().cuda(device_ids[0])
    advLoss = AdvLoss(laplacian_level_count).cuda(device_ids[0])
    d_criterion = nn.CrossEntropyLoss().cuda(device_ids[0])
    g_optimizer = optim.Adam(params=model.parameters(), lr=init_lr, weight_decay=weight_decay)
    d_optimizer = optim.Adam(params=discriminator.parameters(), lr=discriminator_init_lr, weight_decay=discriminator_weight_decay)
    g_lr_sch = optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=epochs, eta_min=final_lr)
    d_lr_sch = optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max=epochs - begin_use_adv_loss_epoch + 1, eta_min=discriminator_final_lr)
    for e in range(epochs):
        current_epoch = e + 1
        train_loader = make_loader(True, train_data_dir, image_size, color_jitter_brightness, color_jitter_saturation, batch_size, laplacian_level_count, num_workers)
        valid_loader = make_loader(False, valid_data_dir, image_size, color_jitter_brightness, color_jitter_saturation, batch_size, laplacian_level_count, num_workers)
        model, discriminator = train_epoch(model, discriminator, recLoss, pyrLoss, advLoss, d_criterion, train_loader, current_epoch, begin_use_adv_loss_epoch, g_optimizer, d_optimizer, device_ids, epochs)
        model, discriminator = valid_epoch(model, discriminator, recLoss, pyrLoss, advLoss, valid_loader, current_epoch, begin_use_adv_loss_epoch, device_ids)
        g_lr_sch.step()
        if current_epoch >= begin_use_adv_loss_epoch:
            d_lr_sch.step()


if __name__ == "__main__":
    conf = json.load(open("conf.json", "r", encoding="utf-8"))
    train_conf = conf["train"]
    CUDA_VISIBLE_DEVICES = train_conf["CUDA_VISIBLE_DEVICES"]
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    device_ids = list(range(len(CUDA_VISIBLE_DEVICES.split(","))))
    best_psnr = -float("inf")
    image_size = train_conf["image_size"]
    discriminator_image_size = train_conf["discriminator_image_size"]
    train_data_dir = train_conf["train_data_dir"]
    valid_data_dir = train_conf["valid_data_dir"]
    batch_size = train_conf["batch_size"]
    init_lr = train_conf["init_lr"]
    final_lr = train_conf["final_lr"]
    discriminator_init_lr = train_conf["discriminator_init_lr"]
    discriminator_final_lr = train_conf["discriminator_final_lr"]
    epochs = train_conf["epochs"]
    begin_use_adv_loss_epoch = train_conf["begin_use_adv_loss_epoch"]
    num_workers = train_conf["num_workers"]
    model_save_dir = train_conf["model_save_dir"]
    laplacian_level_count = train_conf["laplacian_level_count"]
    layer_count_of_every_unet = train_conf["layer_count_of_every_unet"]
    first_layer_out_channels_of_every_unet = train_conf["first_layer_out_channels_of_every_unet"]
    color_jitter_brightness = train_conf["color_jitter_brightness"]
    color_jitter_saturation = train_conf["color_jitter_saturation"]
    weight_decay = train_conf["weight_decay"]
    discriminator_weight_decay = train_conf["discriminator_weight_decay"]
    main()