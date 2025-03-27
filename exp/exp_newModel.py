import os
import time
import warnings

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, visual, LargeScheduler, attn_map, adjust_learning_rate, visual_multi
from utils.losses import forecast_backcast_qt_loss
warnings.filterwarnings('ignore')


class Exp_newModel_forecast(Exp_Basic):

    def _build_model(self):
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = self.model_dict[self.args.model].Model(self.args)
            model = DDP(model.cuda(), device_ids=[self.args.local_rank], find_unused_parameters=False)
        else:
            self.args.device = self.device
            model = self.model_dict[self.args.model].Model(self.args)
        if self.args.is_finetuning:
            checkpoint_path = os.path.join(self.args.checkpoints,self.args.ckpt_path,'checkpoint.pth')
            model.load_state_dict(torch.load(checkpoint_path))
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.use_weight_decay:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = forecast_backcast_qt_loss(self.args.quantilies, self.args.quantile_flag)
        return criterion

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=-5)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        print(('Start training: Training data length: {}, Validation data length: {};' +
               '{} model args: e_layers: {}, d_ff: {}, d_model: {}')
              .format(len(train_data), len(vali_data), self.model.__class__.__name__, self.args.e_layers,
                      self.args.d_ff, self.args.d_model))

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss_list = []

            self.model.train()
            epoch_time = time.time()
            print("Epoch: {}/{} starts, estimated cost time: {}".format(epoch + 1, self.args.train_epochs,
                                                                        time.time() - epoch_time))
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                # batch_x_mark = batch_x_mark.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                # batch_y_mark = batch_y_mark.float().to(self.device)
                
                # decoder input
                #dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                #dec_inp = torch.cat([batch_y[:, :self.args.seq_len, :], dec_inp], dim=1).float().to(self.device)
                #dec_inp_mark = None

                outputs, backcast = self.model(batch_x, None, None, None) # [B, T, Q, M], [B, T, M]
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :, f_dim:]
                backcast = backcast[:, :self.args.seq_len, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                # batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss_value = criterion(outputs, batch_y, backcast, batch_x)
                #loss_sharpness = mse((outputs[:, 1:, :] - outputs[:, :-1, :]), (batch_y[:, 1:, :] - batch_y[:, :-1, :]))
                train_loss = loss_value  # + loss_sharpness * 1e-5
                train_loss_list.append(train_loss.item())


                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, train_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                train_loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss_list)
            vali_loss = self.vali(vali_data, vali_loader, criterion=criterion)
            print("Epoch: {0}/{1} ends | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, self.args.train_epochs, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
        print("Training finished,  best validation loss: {}"
              .format(early_stopping.val_loss_min))

        # best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path, map_location = 'cuda:0'))
        torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')

        return self.model
    
    def vali(self, vali_data, vali_loader, criterion, epoch=0, flag='vali'):
        total_loss = []
        total_count = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # batch_x_mark = batch_x_mark.float().to(self.device)
                # batch_y_mark = batch_y_mark.float().to(self.device)

                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
                if self.args.output_attention:
                    # output used to calculate loss misaligned patch_len compared to input
                    # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    outputs, _ = self.model(batch_x, None, None, None)[0]
                else:
                    # only use the forecast window to calculate loss
                    # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs, _ = self.model(batch_x, None, None, None)
                
                pred_true_idx = self.args.quantilies.index(0.5)
                accurate_outputs = outputs[:, :, pred_true_idx, :]
                if self.args.use_ims:
                    pred = accurate_outputs[:, -self.args.seq_len:, :]
                    true = batch_y
                    if flag == 'vali':
                        loss = self.vali_loss_func(pred, true)
                    elif flag == 'test':  # in this case, only pred_len is used to calculate loss
                        pred = pred[:, -self.args.pred_len:, :]
                        true = true[:, -self.args.pred_len:, :]
                        loss = self.vali_loss_func(pred, true)
                else:
                    loss = self.vali_loss_func(accurate_outputs[:, -self.args.pred_len:, :], batch_y[:, -self.args.pred_len:, :])

                loss = loss.detach().cpu()
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])
                torch.cuda.empty_cache()

        if self.args.use_multi_gpu:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average(total_loss, weights=total_count)
        self.model.train()
        return total_loss

    def test(self, setting, test=0):

        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        attns = []
        folder_path = '../../../data/newModel/test_results/' + setting + '/' + self.args.data_path + '/' + f'{self.args.output_len}/'
        if not os.path.exists(folder_path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(folder_path)
        self.model.eval()
        if self.args.output_len_list is None:
            self.args.output_len_list = [self.args.output_len]

        # preds_list length: 1
        preds_list = [[] for _ in range(len(self.args.output_len_list))]
        trues_list = [[] for _ in range(len(self.args.output_len_list))]
        self.args.output_len_list.sort()

        with torch.no_grad():
            for output_ptr in range(len(self.args.output_len_list)):
                self.args.output_len = self.args.output_len_list[output_ptr]
                test_data, test_loader = data_provider(self.args, flag='test')
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    batch_x_mark = None
                    batch_y_mark = None
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # inference_steps: autoregressive generation number
                    inference_steps = self.args.output_len // self.args.pred_len
                    dis = self.args.output_len - inference_steps * self.args.pred_len
                    if dis != 0:
                        inference_steps += 1
                    pred_y = []
                    pred_true_idx = self.args.quantilies.index(0.5)
                    for j in range(inference_steps):
                        if len(pred_y) != 0:
                            batch_x = torch.cat([batch_x[:, self.args.pred_len:, :], pred_y[-1]], dim=1)
                            # tmp = batch_y_mark[:, j - 1:j, :]
                            # batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)

                        if self.args.output_attention:
                            outputs, attns, backcast = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            outputs_quantile = outputs
                            
                        else:
                            outputs, backcast = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            outputs_quantile = outputs
                        
                        f_dim = -1 if self.args.features == 'MS' else 0
                        pred_y.append(outputs[:, -self.args.pred_len:, pred_true_idx, :])
                    pred_y = torch.cat(pred_y, dim=1)

                    if dis != 0:
                        pred_y = pred_y[:, :-self.args.pred_len+dis, :, :]

                    if self.args.use_ims:
                        batch_y = batch_y[:, self.args.label_len:self.args.label_len + self.args.output_len, :].to(
                            self.device)
                    else:
                        batch_y = batch_y[:, :self.args.output_len, :].to(self.device)
                    outputs = pred_y.detach().cpu()
                    batch_y = batch_y.detach().cpu()
                    outputs_quantile = outputs_quantile.detach().cpu()
                    

                    if test_data.scale and self.args.inverse:
                        shape = outputs.shape
                        outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                    backcast = backcast.detach().cpu().numpy()
                    outputs_quantile = outputs_quantile.detach().cpu().numpy()

                    quantile_pred_list = []
                    for idx in self.args.quantilies:
                        quantile_idx = self.args.quantilies.index(idx)
                        quantile_pred = outputs_quantile[:, :, quantile_idx, :]
                        quantile_pred_list.append(quantile_pred)

                    outputs = outputs[:, :, f_dim:]
                    backcast = backcast[:, :, f_dim:]
                    batch_y = batch_y[:, :, f_dim:]

                    pred = outputs
                    true = batch_y

                    preds_list[output_ptr].append(pred)
                    trues_list[output_ptr].append(true)
                    if i % 10 == 0:
                        input = batch_x.detach().cpu().numpy()
                        gt = true[0, 0:500, -1]
                        pd = pred[0, 0:500, -1]
                        bc = backcast[0, 0:500, -1]
                        quantile_pred_list = [quantile_pred[0, 0:500, -1] for quantile_pred in quantile_pred_list]
                        if self.args.local_rank == 0:
                            if self.args.output_attention:
                                attn = attns[0].cpu().numpy()[0, 0, :, :]
                                attn_map(attn, os.path.join(folder_path, f'attn_{i}_{self.args.local_rank}.pdf'))

                            visual(gt, pd, os.path.join(folder_path, f'{i}_{self.args.local_rank}.pdf'))
                            
                            visual_multi(gt, quantile_pred_list, os.path.join(folder_path, f'multi_{i}_{self.args.local_rank}.pdf'))
                            
                            

                            if self.args.output_interpretability:
                                visual(gt, bc, os.path.join(folder_path, f'backcast_{i}_{self.args.local_rank}.pdf'))
                                

        if self.args.output_len_list is not None:
            for i in range(len(preds_list)):
                preds = preds_list[i]
                trues = trues_list[i]
                preds = torch.cat(preds, dim=0).numpy()
                trues = torch.cat(trues, dim=0).numpy()
                mae, mse, rmse, mape, mspe = metric(preds, trues)
                print(f"output_len: {self.args.output_len_list[i]}")

                print('mse:{}, mae:{}'.format(mse, mae))
                f = open("result_long_term_forecast.txt", 'a')
                f.write(setting + "  \n")
                f.write('mse:{}, mae:{}'.format(mse, mae))
                f.write('\n')
                f.write('\n')
                f.close()

        return

    def finetune(self, setting):
        finetune_data, finetune_loader = data_provider(self.args, flag='train')
        vali_data, vali_loader = data_provider(self.args, flag='val')
        test_data, test_loader = data_provider(self.args, flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(finetune_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        scheduler = LargeScheduler(self.args, model_optim)


        for epoch in range(self.args.finetune_epochs):
            iter_count = 0

            loss_val = torch.tensor(0., device="cuda")
            count = torch.tensor(0., device="cuda")

            self.model.train()
            epoch_time = time.time()

            print("Step number per epoch: ", len(finetune_loader))
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(finetune_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.output_attention:
                    outputs, attns, backcast = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs, backcast = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if self.args.use_ims:
                    # output used to calculate loss misaligned patch_len compared to input
                    loss = criterion(outputs[:, -self.args.seq_len:, :], batch_y, backcast, batch_x)
                else:
                    # only use the forecast window to calculate loss
                    loss = criterion(outputs[:, -self.args.pred_len:, :], batch_y[:, -self.args.pred_len:, :], backcast, batch_x)

                loss_val += loss
                count += 1

                if i % 50 == 0:
                    cost_time = time.time() - time_now
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f} | cost_time: {3:.0f} | memory: allocated {4:.0f}MB, reserved {5:.0f}MB, cached {6:.0f}MB "
                        .format(i, epoch + 1, loss.item(), cost_time,
                                torch.cuda.memory_allocated() / 1024 / 1024,
                                torch.cuda.memory_reserved() / 1024 / 1024,
                                torch.cuda.memory_cached() / 1024 / 1024))
                    time_now = time.time()

                loss.backward()
                model_optim.step()
                torch.cuda.empty_cache()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if self.args.use_multi_gpu:
                dist.barrier()
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            train_loss = loss_val.item() / count.item()

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            if self.args.train_test:
                test_loss = self.vali(test_data, test_loader, criterion, flag='test')
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))


            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            scheduler.schedule_epoch(epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.use_multi_gpu:
            dist.barrier()
        self.model.load_state_dict(torch.load(best_model_path))


        seasonal_params = self.model.module.backbone.encoder.interpreter_layers[0].state_dict()
        torch.save(seasonal_params, path + '/' + 'seasonal_params.pth')
        trend_params = self.model.module.backbone.encoder.interpreter_layers[1].state_dict()
        torch.save(trend_params, path + '/' + 'trend_params.pth')
        
        encoder_params = self.model.module.backbone.encoder.state_dict()
        torch.save(encoder_params, path + '/' + 'encoder_params.pth')

        enc_proj_params = self.model.module.backbone.enc_proj.state_dict()
        torch.save(enc_proj_params, path + '/' + 'enc_proj_params.pth')

        backcast_proj_params = self.model.module.backbone.backcast_proj.state_dict()
        torch.save(backcast_proj_params, path + '/' + 'backcast_proj_params.pth')

        enc_embedding_params = self.model.module.backbone.patch_embedding.state_dict()
        torch.save(enc_embedding_params, path + '/' + 'enc_embedding_params.pth')





        for name, param in self.model.named_parameters():
            print(name)

        return self.model