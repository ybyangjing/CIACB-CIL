import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import FOSTERNet
from convs.cifar_resnet import resnet32
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
import time
import torch.utils.checkpoint as cp


EPSILON = 1e-8


class FOSTER(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = FOSTERNet(args['convnet_type'], False)
        self._snet = None
        self.beta1 = args["beta1"]
        self.beta2 = args["beta2"]
        self.per_cls_weights = None
        self.is_teacher_wa = args["is_teacher_wa"]
        self.is_student_wa = args["is_student_wa"]
        self.lambda_okd = args["lambda_okd"]
        self.wa_value = args["wa_value"]
        self.oofc = args["oofc"].lower()
        self.a = args["increment"]
        self.dataset = args["dataset"]

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        if self._cur_task > 1:
            self._network = self._snet
        self._total_classes = self._known_classes + \
            data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        self._network_module_ptr = self._network
        logging.info(
            'Learning on {}-{}'.format(self._known_classes, self._total_classes))

        if self._cur_task > 0:
            for p in self._network.convnets[0].parameters():
                p.requires_grad = False
            for p in self._network.oldfc.parameters():
                p.requires_grad = False

        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(
            count_parameters(self._network, True)))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["num_workers"], pin_memory=True)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"])

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def train(self):
        self._network_module_ptr.train()
        self._network_module_ptr.convnets[-1].train()
        if self._cur_task >= 1:
            self._network_module_ptr.convnets[0].eval()

    def _train(self, train_loader, test_loader):
        start_time = time.time()
        print("start time:", start_time)
        timees = 0
        self._network.to(self._device)
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module
        if self._cur_task == 0:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters(
            )), momentum=0.9, lr=self.args["init_lr"], weight_decay=self.args["init_weight_decay"])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args["init_epochs"])
            self._init_train(train_loader, test_loader, optimizer, scheduler)
            mid_time = time.time()
            logging.info("training time: {}".format(mid_time - start_time))
        else:

            if isinstance(self.beta1,list):
                beta1=self.beta1[self._cur_task-1]
            else:
                beta1=self.beta1
            cls_num_list = [self.samples_old_class]*self._known_classes+[
                self.samples_new_class(i) for i in range(self._known_classes, self._total_classes)]
            effective_num = 1.0 - np.power(beta1, cls_num_list)

            per_cls_weights = (1.0 - beta1) / np.array(effective_num)
            for i in range(len(cls_num_list)):
                if i < len(cls_num_list) - self.a:
                    per_cls_weights[i] = per_cls_weights[i] * 1.1
                else:
                    per_cls_weights[i] = per_cls_weights[i] * 1
            per_cls_weights = per_cls_weights / \
                np.sum(per_cls_weights) * len(cls_num_list)

            logging.info("per cls weights : {}".format(per_cls_weights))
            self.per_cls_weights = torch.FloatTensor(
                per_cls_weights).to(self._device)

            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters(
            )), lr=self.args["lr"], momentum=0.9, weight_decay=self.args["weight_decay"])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args["boosting_epochs"])
            if self.oofc == "az":
                for i, p in enumerate(self._network_module_ptr.fc.parameters()):
                    if i == 0:
                        p.data[self._known_classes:, :self._network_module_ptr.out_dim] = torch.tensor(
                            0.0)
            elif self.oofc != "ft":
                assert 0, "not implemented"
            self._feature_boosting(
                train_loader, test_loader, optimizer, scheduler)
            if self.is_teacher_wa:
                self._network_module_ptr.weight_align(
                    self._known_classes, self._total_classes-self._known_classes, self.wa_value)
            else:
                logging.info("do not weight align teacher!")

            cls_num_list = [self.samples_old_class]*self._known_classes+[
                self.samples_new_class(i) for i in range(self._known_classes, self._total_classes)]
            effective_num = 1.0 - np.power(self.beta2, cls_num_list)
            per_cls_weights = (1.0 - self.beta2) / np.array(effective_num)
            for i in range(len(cls_num_list)):
                if i < len(cls_num_list) - self.a:
                    per_cls_weights[i] = per_cls_weights[i] * 1.1
                else:
                    per_cls_weights[i] = per_cls_weights[i] * 1
            per_cls_weights = per_cls_weights / \
                np.sum(per_cls_weights) * len(cls_num_list)
            logging.info("per cls weights : {}".format(per_cls_weights))
            self.per_cls_weights = torch.FloatTensor(
                per_cls_weights).to(self._device)
            self._feature_compression(train_loader, test_loader)
            last_time = time.time()
            times = last_time - start_time
            print("testing time:", last_time - start_time)
            timees += times
            logging.info("timees : {}".format(timees))

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epochs"]))
        self.gap = nn.AdaptiveAvgPool2d((1, 1)).to(self._device)
        self.up_classifier = nn.Sequential(
            nn.Linear(512,100),
        ).to(self._device)
        self.backbone = Non_Local_VGG16().to(self._device)

        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.
            loss1es = 0.
            correct, total = 0, 0
            cls_criterion = nn.CrossEntropyLoss().to(self._device)
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                logits = self._network(inputs)['logits']

                loss = F.cross_entropy(logits, targets.long())
                optimizer.zero_grad()
  
                loss.backward(retain_graph=True)
                optimizer.step()
                losses += loss.item()
              #  loss1es += loss1.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(
                correct)*100 / total, decimals=2)
            if epoch % 5 != 0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss1 {:.3f},Train_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self.args["init_epochs"], losses/len(train_loader), losses/len(train_loader), train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self.args["init_epochs"], losses/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)
            logging.info(info)

    def _feature_boosting(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["boosting_epochs"]))
 
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.
            losses_clf = 0.
            losses_fe = 0.
            losses_kd = 0.
           # cls_criterion = nn.CrossEntropyLoss().to(self._device)
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                outputs = self._network(inputs)
                logits, fe_logits, old_logits = outputs["logits"], outputs["fe_logits"], outputs["old_logits"].detach(
                )
                new_class_weight = 1.3
                old_class_weight = 0.7

                loss_clf = F.cross_entropy(
                    logits/self.per_cls_weights, targets.long())
                loss_clf = (loss_clf * (targets < self._known_classes).float() * old_class_weight +
                    loss_clf * (targets >= self._known_classes).float() * new_class_weight).mean()

                fe_targets = targets.clone()
                if self._known_classes == 5:
                    fe_targets = torch.where(
                        fe_targets - self._known_classes + 4 > 0,
                        fe_targets - self._known_classes + 4,
                        0,
                    )
                else:
                    fe_targets = torch.where(
                        fe_targets - self._known_classes + 6 > 0,
                        fe_targets - self._known_classes + 6,
                        0,
                    )
               # new_class_weight = 1
               # old_class_weight = 1
                loss_fe = F.cross_entropy(fe_logits* (self.per_cls_weights), fe_targets.long())
              #  loss_fe = (loss_fe * (targets < self._known_classes).float() * old_class_weight +
              #      loss_fe * (targets >= self._known_classes).float() * new_class_weight).mean()
                loss_kd = self.lambda_okd * \
                    _KD_loss(logits[:, :self._known_classes],
                             old_logits, self.args["T"])*0.95
              
                loss = loss_clf+loss_fe+loss_kd 
                 # Add  L2  Regularization
                l2_reg = torch.tensor(0.).to(self._device)
                l2_lambda = 0.0000004
                for param in self._network.parameters():
                    # 通过将L2正则化项添加到损失函数中，可以对模型的权重进行惩罚。L2正则化通过向损失函数添加一个惩罚项，
                    # 使得模型在训练过程中更倾向于使用较小的权重值。这有助于防止模型过度拟合训练数据，使其更具泛化能力。
                    l2_reg += torch.norm(param) ** 2
                loss += l2_lambda * l2_reg
                optimizer.zero_grad()
                loss.backward()
                if self.oofc == "az":
                    for i, p in enumerate(self._network_module_ptr.fc.parameters()):
                        if i == 0:
                            p.grad.data[self._known_classes:, :self._network_module_ptr.out_dim] = torch.tensor(
                                0.0)
                elif self.oofc != "ft":
                    assert 0, "not implemented"
                optimizer.step()
                losses += loss.item()
                losses_fe += loss_fe.item()
                losses_clf += loss_clf.item()
                losses_kd += (self._known_classes /
                              self._total_classes)*loss_kd.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(
                correct)*100 / total, decimals=2)
            if epoch % 5 != 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fe {:.3f}, Loss_kd {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self.args["boosting_epochs"], losses/len(train_loader), losses_clf/len(train_loader), losses_fe/len(train_loader), losses_kd/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fe {:.3f}, Loss_kd {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self.args["boosting_epochs"], losses/len(train_loader), losses_clf/len(train_loader), losses_fe/len(train_loader), losses_kd/len(train_loader), train_acc)
            prog_bar.set_description(info)
            logging.info(info)

    def _feature_compression(self, train_loader, test_loader):
        self._snet = FOSTERNet(self.args['convnet_type'], False)
        self.gap = nn.AdaptiveAvgPool2d((1, 1)).to(self._device)
        if self.dataset  ==  "cifar100" :
                    self.up_classifier = nn.Sequential(
            nn.Linear(128,self._total_classes),
        ).to(self._device)        #5555555555555555555555555555555555555555555555555555555555555555555555555
        else :
                    self.up_classifier = nn.Sequential(
            nn.Linear(1024,self._total_classes),
        ).to(self._device)
        
        self.backbone = Non_Local_VGG16().to(self._device)
        self.mask = torch.zeros(size=[self._total_classes, 14, 14], requires_grad=False).to(self._device)
       # self.mask2attention = nn.Conv2d(self._total_classes, 128, 1, 1, 0).to(self._device)
        if self.dataset  ==  "cifar100" :
                    self.mask2attention = nn.Conv2d(self._total_classes, 128, 1, 1, 0).to(self._device)        #5555555555555555555555555555555555555555555555555555555555555555555555555
        else :
                    self.mask2attention = nn.Conv2d(self._total_classes, 1024, 1, 1, 0).to(self._device)

        self._snet.update_fc(self._total_classes)
        if len(self._multiple_gpus) > 1:
            self._snet = nn.DataParallel(self._snet, self._multiple_gpus)
        if hasattr(self._snet, "module"):
            self._snet_module_ptr = self._snet.module
        else:
            self._snet_module_ptr = self._snet
        self._snet.to(self._device)
        self._snet_module_ptr.convnets[0].load_state_dict(
            self._network_module_ptr.convnets[0].state_dict())
        self._snet_module_ptr.copy_fc(self._network_module_ptr.oldfc)
        optimizer = optim.SGD(filter(
            lambda p: p.requires_grad, self._snet.parameters()), lr=self.args["lr"], momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.args["compression_epochs"])
        self._network.eval()
         
        try:
            context_new = torch.zeros(self._snet.fc.weight.shape, dtype=torch.float32).cuda()
            current_num = torch.zeros(self._snet.fc.weight.shape[0], dtype=torch.float32, device=self._device)
            class_num = self._snet.fc.weight.shape[0]  # 设置了一个变量 class_num，用来存储权重张量的第一维的大小。

        except:
            try:
                context_new = torch.zeros(self._snet.linear.weight.shape, dtype=torch.float32).cuda()
                current_num = torch.zeros(self._snet.linear.weight.shape[0], dtype=torch.float32, device=self._device)
                class_num = self._snet.linear.weight.shape[0]
            except:
                context_new = torch.zeros(self._snet.classifier.weight.shape, dtype=torch.float32).cuda()
                current_num = torch.zeros(self._snet.classifier.weight.shape[0], dtype=torch.float32, device=self._device)
                class_num = self._snet.classifier.weight.shape[0]
        criterion_kd = NORM_MSE()
        criterion_cc = Cosine()
        prog_bar = tqdm(range(self.args["compression_epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._snet.train()
            losses = 0.
            loss1es = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                dark_logits, dark_featss = self._snet(inputs)["logits"], self._snet(inputs)["features"]
                with torch.no_grad():
                    outputs = self._network(inputs)
                    logits, old_logits, fe_logits, feats = outputs["logits"], outputs["old_logits"], outputs[
                        "fe_logits"], outputs["features"]
                Cs_h = dark_featss[-1].detach()
                Ct_h = feats[-1].detach()
                model_s_fc_news = Reg(Cs_h.shape[0] * 2, self._total_classes)
                model_s_fc_new = model_s_fc_news.to(self._device)
                soft_ts = torch.softmax(logits / 12, dim=1)
                soft_t = soft_ts.to(self._device)
                dark_feats = dark_featss.to(self._device).detach()
                for i in range(len(targets)):
                     if i < Cs_h.shape[0]:
                           context_new[targets[i]] = context_new[targets[i]] * (   
                              current_num[targets[i]] /((current_num[targets[i]] + soft_t[i][targets[i]]))) +Cs_h[i]*(soft_t[i][targets[i]] / (current_num[targets[i]] + soft_t[i][targets[i]]))
                     else:
                           context_new[targets[i]] = context_new[targets[i]] * (
                              current_num[targets[i]] / (current_num[targets[i]] + soft_t[i][targets[i]]))
                for i in range(len(targets)):    
                     current_num[targets[i]] = current_num[targets[i]]+ soft_t[i][targets[i]]
                p = torch.softmax(dark_logits / 12, dim=1)

                context = context_new
                sam_contxt = torch.mm(p, context)
                f_new = torch.cat((dark_feats, sam_contxt), 1) / self._total_classes
                                     #85555555555555555555555555555
                if self.dataset  ==  "cifar100" :
                    logit_s_news = model_s_fc_new(f_new[:, :128])         #5555555555555555555555555555555555555555555555555555555555555555555555555
                else :
                    logit_s_news = model_s_fc_new(f_new[:, :1024])         #5555555555555555555555555555555555555555555555555555555555555555555555555
                logit_s_new = logit_s_news.to(self._device)
                logits_copy = logit_s_new.clone()  # 创建 logits 的副本
                targets_copy = targets.clone()  # 创建 targets 的副本
                loss_cls = F.cross_entropy(logits_copy, targets_copy.long())*6
                

                loss_dark =0.5 * self.BKD(dark_logits, logits, self.args["T"])# BKD即平衡蒸馏损失，logits即论文中的Ft（x）、
                loss_clf = F.cross_entropy(dark_logits / self.per_cls_weights, targets.long()) * 0.05
                loss_kd = _KD_loss(
                    dark_logits[:, : self._known_classes], old_logits, self.args["T"]
                ) * 0.5

                features = self._network(inputs)['features']
                featuresss = features.unsqueeze(-1).unsqueeze(-1)
                feature_map = featuresss
             #   print("feature_map shape:", feature_map.shape)
         
                
                
                up_vector = self.gap(feature_map).view(feature_map.size(0), -1)
                self.up_out = self.up_classifier(up_vector)
                self.pred_sort_up, self.pred_ids_up = torch.sort(self.up_out, dim=-1, descending=True)
                self.up_cam = self._compute_cam(feature_map, self.up_classifier[0].weight)   
                self.up_cam = torch.mean(self.up_cam, dim=(2, 3))


                context_list = torch.zeros(size=[self.pred_ids_up.shape[0], self._total_classes, 14, 14],
                                      requires_grad=False).to(self._device)
                source_inds = get_pre_two_source_inds(shap=context_list.shape).to(self._device)
                context_list = context_list[source_inds, self.pred_ids_up]  # reverse
                context_list[:, 0] = self.mask[self.pred_ids_up[:, 0]] / self._total_classes
                context_list = 1  * context_list[source_inds, torch.argsort(self.pred_ids_up, dim=1)].to(self._device) # reverse back
                temp_attention = self.mask2attention(context_list.to(torch.float32))








                
              #  if self.dataset  ==  "cifar100" :
              #      self.mask_bn = nn.BatchNorm2d(128).to(self._device)        #5555555555555555555555555555555555555555555555555555555555555555555555555
              #  else :
              #      self.mask_bn = nn.BatchNorm2d(1024).to(self._device)


              #  tmp_mask = self.mask_bn(temp_attention).to(self._device).detach()
              #  temp_attention = temp_attention + 0.03*tmp_mask
              #  temp_attention = self.mask_bn(temp_attention).to(self._device)












   
                self.down_feature_map = torch.add(feature_map, temp_attention.mul(feature_map))
                down_vector = self.gap(self.down_feature_map).view(self.down_feature_map.size(0), -1)
                self.down_out = self.up_classifier(down_vector)
             #   print("self.down_out shape:", self.down_out)
                self.down_cam = self._compute_cam(self.down_feature_map, self.up_classifier[0].weight)
                self.down_cam = torch.mean(self.down_cam, dim=(2, 3))
                
                loss1 = _KD_loss(
                   self.down_out,dark_logits, self.args["T"]
                ) *0.17 
               # loss2 = F.cross_entropy(self.down_out, targets)*0.5



                loss = loss_dark + loss_cls +  loss1 + loss_kd + loss_clf 
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                loss1es += loss1.item()
                _, preds = torch.max(dark_logits[:targets.shape[0]], dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(
                correct)*100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._snet, test_loader)
                info = 'SNet: Task {}, Epoch {}/{} => Loss {:.3f},  Loss1 {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self.args["compression_epochs"], losses/len(train_loader), loss1es/len(train_loader), train_acc, test_acc)
            else:
                info = 'SNet: Task {}, Epoch {}/{} => Loss {:.3f},  Loss1 {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self.args["compression_epochs"], losses/len(train_loader),  loss1es/len(train_loader), train_acc)
            prog_bar.set_description(info)
            logging.info(info)
        if len(self._multiple_gpus) > 1:
            self._snet = self._snet.module
        if self.is_student_wa:
            self._snet.weight_align(
                self._known_classes, self._total_classes-self._known_classes, self.wa_value)
        else:
            logging.info("do not weight align student!")

        self._snet.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(self._device, non_blocking=True)
            with torch.no_grad():
                outputs = self._snet(inputs)['logits']
            predicts = torch.topk(outputs, k=self.topk,
                                  dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        cnn_accy = self._evaluate(y_pred, y_true)
        logging.info("darknet eval: ")
        logging.info('CNN top1 curve: {}'.format(cnn_accy['top1']))
        logging.info('CNN top5 curve: {}'.format(cnn_accy['top5']))
        top1_values = cnn_accy['top1']

        # 计算平均值
        top1_mean = np.mean(top1_values)

        # 打印平均值
        logging.info('CNN top1 平均值: {:.2f}'.format(top1_mean))

    @property
    def samples_old_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, 'Total classes is 0'
            return (self._memory_size // self._known_classes)

    def samples_new_class(self, index):
        if self.args["dataset"] == "cifar100":
            return 500
        else:
            return self.data_manager.getlen(index)

    def BKD(self, pred, soft, T):
        pred = torch.log_softmax(pred/T, dim=1)
        soft = torch.softmax(soft/T, dim=1)
        soft = soft*self.per_cls_weights
        soft = soft / soft.sum(1)[:, None]
        return -1*torch.mul(soft, pred).sum()/pred.shape[0]


    def update_mask(self, tmp_mask):
        # tensor.detach() no gradient，requires_grad = False.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mask_bn = nn.BatchNorm2d(self._total_classes).to(device)
        tmp_mask = self.mask_bn(tmp_mask).to(device).detach()
        self.mask = torch.add(self.mask, torch.mul(tmp_mask, 0.3)).to(device)
      #  logging.info('Before unsqueeze: self.mask is: {}'.format(self.mask))
        self.mask = self.mask_bn(self.mask).to(device)


    def _compute_cam(self, input, weight):
        """
        :param input:
        :param weight:
        :return:
        """
        input = input.permute(1, 0, 2, 3)
        nc, bz, h, w = input.shape
        input = input.reshape((nc, bz * h * w))
        cams = torch.matmul(weight, input)
        cams = cams.reshape(self._total_classes, bz, h, w)
        cams = cams.permute(1, 0, 2, 3)
        return cams

    def cw_decorrelation_loss(self, features, targets):
        """Class-wise Decorrelation (CwD) Loss"""
        unique_targets = torch.unique(targets)
        loss = 0.0
        for cls in unique_targets:
            cls_indices = (targets == cls).nonzero(as_tuple=True)[0]
            cls_features = features[cls_indices]
            if cls_features.size(0) > 1:
                cls_features = F.normalize(cls_features, p=2, dim=1)
                gram_matrix = torch.matmul(cls_features, cls_features.t())
                ones = torch.ones_like(gram_matrix)
                diag_mask = torch.eye(gram_matrix.size(0), device=gram_matrix.device)
                loss += torch.sum((gram_matrix - diag_mask) ** 2) / (cls_features.size(0) * (cls_features.size(0) - 1))
        return loss / unique_targets.size(0)

def get_pre_two_source_inds(shap):
    bz, nc = shap[0:2]
    return torch.arange(bz).unsqueeze(dim=1).expand((bz, nc))

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred/T, dim=1)
    soft = torch.softmax(soft/T, dim=1)
    return -1*torch.mul(soft, pred).sum()/pred.shape[0]

def KL(pred, soft, T):
    """KL_div"""
    p_s = F.log_softmax(pred/T, dim=1)
    p_t = F.softmax(soft/T, dim=1)
    loss = F.kl_div(p_s, p_t, size_average=False) * (T**2) / pred.shape[0]
    return loss

def kl_divergence_loss(logits_new, logits_old):
    # 对 logits_new 和 logits_old 进行 softmax 操作，使其变成概率分布
    prob_new = F.softmax(logits_old, dim=-1)
    prob_old = F.softmax(logits_new, dim=-1)

    # 确定剪断的位置，使得旧类与新类的大小相同
    num_new_classes = prob_new.size(1)
    num_old_classes = prob_old.size(1)
    min_classes = min(num_new_classes, num_old_classes)

    # 将旧类剪断到新类的大小
    prob_old_cut = prob_old[:, :min_classes]
    prob_new_cut = prob_new[:, :min_classes]

    # 计算 KL 散度
    kl_div = torch.sum(prob_new_cut * (torch.log(prob_new) - torch.log(prob_old_cut)), dim=-1)

    # 返回平均 KL 散度作为损失
    return torch.mean(kl_div)

class Cosine(nn.Module):

    def __init__(self):
        super(Cosine, self).__init__()

    def forward(self, g_s, g_t):
        return self.similarity_loss(g_s, g_t)

    def similarity_loss(self, f_s, f_t):
        bsz = f_s.size(0)  # 64*

        f_s = f_s.view(bsz, -1)  # 64*dim
        f_s = torch.nn.functional.normalize(f_s)  # 64*dim

        f_t = f_t.view(bsz, -1)
        f_t = torch.nn.functional.normalize(f_t)  # 64*dim

        G_s = torch.mm(f_s, torch.t(f_s))  # 64*dim
        # G_s = G_s / G_s.norm(2)

        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)

        G_diff = G_t - G_s

        # print('G_diff0', G_diff)

        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)

        return loss


class NORM_MSE(nn.Module):

    def __init__(self):
        super(NORM_MSE, self).__init__()
        self.MS = nn.MSELoss(reduction='none')  # nn.MSELoss(size_average=False)

    def forward(self, output, target):
        target = target.view(target.shape[0], -1)

        output = output.view(output.shape[0], -1)

        magnitute = torch.norm(target, dim=1)

        magnitute_square = magnitute ** 2

        magnitute_square = torch.reshape(magnitute_square, (output.shape[0], -1))

        loss = torch.sum(self.MS(output, target) / magnitute_square) / target.shape[0]

        return loss


class Reg(nn.Module):
    """Simple Linear Regression for hints"""    #简单线性回归提示
    def __init__(self, dim_in=1024, dim_out=1024):
        super(Reg, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = self.linear(x)
        return x


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()
        self.inter_channels = in_channels // 2 if in_channels > 1 else 1

        self.g = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.W = nn.Conv2d(self.inter_channels, in_channels, 1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        phi_x = phi_x.permute(0, 2, 1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class Non_Local_VGG16(nn.Module):
    def __init__(self, pretrain=True, num_classes=100):
        super(Non_Local_VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # 特征正则化层，用于减少遗忘
        self.regularizer = nn.Linear(512 * 7 * 7, num_classes)

    def forward(self, x):
        #x = self.features(x)
        #x = self.avgpool(x)
      # x = x.view(x.size(0), -1)
      #  x = self.regularizer(x)
        return self.features(x)

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class DecorrelateLossClass(nn.Module):

    def __init__(self, reject_threshold=1, ddp=False):
        super(DecorrelateLossClass, self).__init__()
        self.eps = 1e-8
        self.reject_threshold = reject_threshold
        self.ddp = ddp

    def forward(self, x, y):
        _, C = x.shape
        if self.ddp:
            # if DDP
            # first gather all x and labels from the world
            x = torch.cat(GatherLayer.apply(x), dim=0)
            y = global_gather(y)

        loss = 0.0
        uniq_l, uniq_c = y.unique(return_counts=True)
        n_count = 0
        for i, label in enumerate(uniq_l):
            if uniq_c[i] <= self.reject_threshold:
                continue
            x_label = x[y==label, :]
            x_label = x_label - x_label.mean(dim=0, keepdim=True)
            x_label = x_label / torch.sqrt(self.eps + x_label.var(dim=0, keepdim=True))

            N = x_label.shape[0]
            corr_mat = torch.matmul(x_label.t(), x_label)

            # Notice that here the implementation is a little bit different
            # from the paper as we extract only the off-diagonal terms for regularization.
            # Mathematically, these two are the same thing since diagonal terms are all constant 1.
            # However, we find that this implementation is more numerically stable.
            loss += (off_diagonal(corr_mat).pow(2)).mean()

            n_count += N

        if n_count == 0:
            # there is no effective class to compute correlation matrix
            return 0
        else:
            loss = loss / n_count
            return loss