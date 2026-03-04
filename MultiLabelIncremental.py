"""
Multi-Label Incremental
"""
import os
import time
import torch
import math
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from randaugment import RandAugment
from copy import deepcopy

from src.helper_functions.utils import build_logger, calculate_metrics, print_to_excel, get_percentile, non_linear_transform
from src.helper_functions.helper_functions import CutoutPIL, AverageMeter, add_weight_decay, mAP
from src.helper_functions.IncrementalDataset import build_dataset, build_loader
from src.helper_functions.coco_loader import coco_fake2real, coco_ids_to_cats
from src.loss_functions.distillation import pod, embeddings_similarity
from src.loss_functions.losses import AsymmetricLoss, DistillationLoss, MultiLabelDistillationLoss
from src.models import create_model
from sample_proto import sample_protos, sample_protos_buffer


class MultiLabelIncremental(object):
    def __init__(self, args):
        """
        Initialize Multi-Label Incremental Object
        """
        self.args = args
        
        # Logger
        self.log_frequency = 100
        self.logger = build_logger(args.logger_dir)
        self.logger.info('Running Multi-Label Incremental Learning!')
        self.logger.info('Arguments:')
        for k, v in sorted(vars(args).items()):
            self.logger.info('{}={}'.format(k, v))

        # Model Save Path
        self.model_save_path = args.model_save_path
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # Excel Path
        self.excel_path = args.excel_path

        # Train Parameters
        self.nb_epochs = args.epochs
        self.incr_lr = args.incre_lr
        self.base_lr = args.base_lr
        self.weight_decay = args.weight_decay

        # Incremental Setups
        self.total_classes = args.total_classes
        self.base_classes = args.base_classes
        self.task_size = args.task_size
        self.num_classes = self.base_classes

        # Knowledge Distillation
        self.kd_loss_choice = args.kd_loss_choice
        self.lambda_c = args.lambda_c
        self.lambda_f = args.lambda_f

        # Model
        self.model_name = args.model_name
        self.pretrained_path = args.pretrained_path
        self.old_model = None
        self.model = self.setup_model()

        # Datasets
        self.dataset_name = args.dataset_name
        self.root_dir = args.root_dir
        self.image_size = args.image_size
        self.train_batch_size = args.train_batch_size
        self.val_batch_size = args.val_batch_size
        self.num_workers = args.num_workers
        self.train_transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            CutoutPIL(cutout_factor=0.5),
            RandAugment(),
            transforms.ToTensor()
        ])
        self.val_transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])

        # Replay
        self.replay = args.replay
        self.num_protos = args.num_protos
        self.fix_budget = args.fix_budget
        self.buffer_size = args.buffer_size
        self.buffer_old_threshold = args.buffer_old_threshold
        self.buffer_new_threshold = args.buffer_new_threshold
        self.threshold_sigma = args.threshold_sigma
        self.min_threshold_sigma = args.min_threshold_sigma
        if self.fix_budget:
            self.old_dataset = buffer(self.buffer_size, self.buffer_old_threshold, self.buffer_new_threshold)
        else:
            self.old_dataset = []

         # Pseudo Label
        self.pseudo_training = args.pseudo_training
        self.pseudo_buffer = args.pseudo_buffer
        self.label_buffer = args.label_buffer
        self.threshold = args.threshold


    def setup_model(self):
        """
        Create Model: Load Pre-Trained Weight
        """
        assert self.pretrained_path
        model = create_model(self.args, self.base_classes)
        model.cuda()
        state = torch.load(self.pretrained_path, map_location='cpu')
        state = {(k if 'body.' not in k else k[5:]): v for k, v in state['model'].items()}
        filtered_dict = {k: v for k, v in state.items() if
                                (k in model.state_dict() and 'head.fc' not in k)}
        model.load_state_dict(filtered_dict, strict=False)
        self.logger.info(f'Create Model Successfully. Loaded from Model_Path:{self.pretrained_path}. Loaded Params:{len(filtered_dict)}. \n')
        return model
    

    def train(self):
        """ 
        Train Function 
        """
        ########## 
        # Metric #
        ##########
        mAP_meter = AverageMeter()
        mAP_list = np.zeros((self.total_classes-self.base_classes) // self.task_size + 1)
        
        ############### 
        # Incremental #
        ###############
        base_stage = [(0, self.base_classes)]
        incremental_stages = base_stage + [
                (low, low + self.task_size) for low in range(self.base_classes, self.total_classes, self.task_size)]
        
        for low_range, high_range in incremental_stages:
            # Before Task
            self._before_task(low_range)
            self.num_classes = high_range
            # Get Dataset
            train_dataset_without_old = build_dataset(self.dataset_name, 
                                                      self.root_dir, 
                                                      low_range, 
                                                      high_range,
                                                      phase='train', 
                                                      transform=self.train_transforms
                                                      )
            self.logger.info(f'Current Incremental Stage: ({low_range}, {high_range}), Dataset Length:{len(train_dataset_without_old)}')
            train_dataset = train_dataset_without_old

            if self.replay and self.old_dataset and low_range != 0:
                train_dataset_with_old = [train_dataset]
                if self.fix_budget:
                    train_dataset_with_old.append(self.old_dataset.get_bufferset())
                    train_dataset_with_old = torch.utils.data.ConcatDataset(train_dataset_with_old)
                else:
                    train_dataset_with_old.extend(self.old_dataset)
                    train_dataset_with_old = torch.utils.data.ConcatDataset(train_dataset_with_old)

                self.logger.info(f'Current Incremental Stage: ({low_range}, {high_range}), Dataset with Old Samples Length: {len(train_dataset_with_old)}')
                train_dataset = train_dataset_with_old

            # Validation Datasets: Seen
            val_dataset_seen = build_dataset(self.dataset_name, 
                                             self.root_dir, 
                                             0, 
                                             high_range, 
                                             phase='val',
                                             transform=self.val_transforms
                                             )
            
            # Build Loaders
            train_loader = build_loader(train_dataset, self.train_batch_size, self.num_workers, phase='train')
            val_loader_seen = build_loader(val_dataset_seen, self.train_batch_size, self.num_workers, phase='val')

            # Training Process
            mAP, metrics = self._train_task(self.nb_epochs, 
                                            low_range, 
                                            high_range, 
                                            train_loader, 
                                            val_loader_seen)
            mAP_meter.update(mAP)
            mAP_list[(high_range-self.base_classes)//self.task_size] = mAP

            # After task
            if self.fix_budget:
                self._after_task(low_range, high_range, train_dataset=train_dataset_without_old, buffer_dataset=self.old_dataset.get_bufferset())
            else:
                if self.old_dataset:
                    self._after_task(low_range, high_range, train_dataset=train_dataset_without_old, buffer_dataset=torch.utils.data.ConcatDataset(self.old_dataset))
                else:
                    self._after_task(low_range, high_range, train_dataset=train_dataset_without_old, buffer_dataset=None)

            # Save Results
            if 'coco' in self.dataset_name:
                ds_name = 'COCO'
            if 'voc' in self.dataset_name:
                ds_name = 'VOC'
            params = f"LR: {self.lr}, epoch: {self.nb_epochs}, BS: {self.train_batch_size}, Replay: {self.replay}, Fix_Budget: {self.fix_budget}, KDLoss: {self.kd_loss_choice}, Pseudo_Training: {self.pseudo_training}, Pseudo_Buffer: {self.pseudo_buffer}, Protos: {self.num_protos}"
            print_to_excel(self.excel_path, 
                           self.args.output_name, 
                           ds_name, 
                           self.base_classes, 
                           self.task_size, 
                           self.total_classes, 
                           params, 
                           mAP_list, 
                           metrics)
        del self.logger


    def _before_task(self, low_range):
        self.model.eval()
        # Change the Fully Connected Layer
        if low_range != 0:
            in_dimension = self.model.head.fc.in_features
            old_classes = self.model.head.fc.out_features

            new_fc = nn.Linear(in_dimension, old_classes + self.task_size)
            new_fc.weight.data[:old_classes] = self.model.head.fc.weight.data
            new_fc.bias.data[:old_classes] = self.model.head.fc.bias.data
            new_fc.cuda()

            self.model.head.fc = new_fc


    def _train_task(self, 
                    nb_epochs, 
                    low_range, 
                    high_range, 
                    train_loader,
                    val_loader_seen):
        self.model.train()
        
        if low_range == 0:
            self.lr = self.base_lr
        else:
            self.lr = self.incr_lr
        parameters = add_weight_decay(self.model, self.weight_decay)
        self.cls_criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
        self.optimizer = torch.optim.Adam(params=parameters, lr=self.lr, weight_decay=0)
        self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, 
                                                 max_lr=self.lr, 
                                                 steps_per_epoch=len(train_loader),
                                                 epochs=self.nb_epochs,
                                                 pct_start=0.2)

        scaler = GradScaler()

        buffer_indices = None
        if low_range != 0 and self.pseudo_buffer:
            buffer_indices = []
            for i, (_, target) in enumerate(train_loader):
                mask = (target[:, low_range:high_range] < 1).all(dim=1)
                buffer_indices.append(torch.where(mask)[0] + i * self.train_batch_size)
            buffer_indices = torch.cat(buffer_indices)

        for epoch in range(nb_epochs):
            epoch_start = time.time()
            self._train_one_epoch(train_loader, scaler, low_range, high_range, epoch, buffer_indices)
            train_epoch_time = time.time() - epoch_start
            self.logger.info(f'Train One Epoch Time: {train_epoch_time:.2f}')

            val_start = time.time()
            self.logger.info("Start Validation")
            self.model.eval()
            val_result, val_result2 = self.validate_seen(high_range, val_loader_seen)
            self.logger.info(f"current_mAP_seen = {val_result[0]:.2f}")
            self.logger.info('current other metrics: mean_p_c: {}, mean_r_c: {}, mean_f_c: {}, precision_o: {}, recall_o: {}, f1_o: {}'
                              .format(*[item for item in val_result2]))
            self.logger.info(f'Validation_time: {time.time() - val_start:.2f}')

        return val_result[0], val_result2


    def _train_one_epoch(self, 
                         train_loader, 
                         scaler, 
                         low_range, 
                         high_range, 
                         epoch,
                         buffer_indices):
        self.model.train()
        self.model.zero_grad(set_to_none=True)

        # Calculate the Threshold for Different Classes at Each Time Step
        cls_thresholds = torch.tensor([self.threshold_sigma] * (high_range - low_range))
        count_list = [0] * (high_range - low_range)
        if low_range != 0 and self.pseudo_buffer and epoch > 0:
            for i, (image, target) in enumerate(train_loader):
                start_idx = i * self.train_batch_size
                end_idx = (i + 1) * self.train_batch_size
                batch_mask = (buffer_indices >= start_idx) & (buffer_indices < end_idx)
                batch_indices = buffer_indices[batch_mask] - start_idx

                if len(batch_indices) > 0:
                    image = image.cuda(non_blocking=True)
                    target = target[:, :high_range]
                    with torch.no_grad():
                        with autocast():
                            output = self.model(image)
                            new_logits = torch.sigmoid(output['logits']).cpu()
                            del image, output
                            torch.cuda.empty_cache()
                    logits_to_compare = new_logits[batch_indices, low_range:high_range]
                    expanded_thresholds = cls_thresholds.unsqueeze(0).expand(logits_to_compare.shape)
                    predictions = logits_to_compare > expanded_thresholds
                    prediction_counts = predictions.sum(dim=0)
                    for i, count in enumerate(prediction_counts):
                        count_list[i] += count.item()
            count_tensor = torch.tensor(count_list)
            sum_count = count_tensor.sum()
            normalized_counts = count_tensor / sum_count
            cls_thresholds = torch.exp(-normalized_counts)
            cls_thresholds = torch.maximum(cls_thresholds, torch.tensor(self.min_threshold_sigma)).cuda(non_blocking=True)

        # Training
        for i, (image, target) in enumerate(train_loader):
            image = image.cuda(non_blocking=True)
            target = target[:, :high_range].cuda(non_blocking=True)

            old_output = None
            if self.old_model and (self.pseudo_training or self.kd_loss_choice):
                with torch.no_grad():
                    old_output = self.old_model(image)
                    if self.pseudo_training:
                        old_logits = torch.sigmoid(old_output['logits'])
                        new_data_mask = (target[:, :low_range] < 1).all(dim=1)
                        pseudo_label_mask = (old_logits[:, :low_range] > self.threshold) & new_data_mask.unsqueeze(1)
                        target[:, :low_range][pseudo_label_mask] = 1
                    
                    torch.cuda.empty_cache()
            
            with autocast():
                output = self.model(image)
                # Pseudo Buffer
                if low_range != 0 and self.pseudo_buffer and epoch > 0:
                    start_idx = i * train_loader.batch_size
                    end_idx = (i + 1) * train_loader.batch_size
                    batch_mask = (buffer_indices >= start_idx) & (buffer_indices < end_idx)
                    batch_indices = buffer_indices[batch_mask] - start_idx
                    if len(batch_indices) > 0:
                        new_logits = torch.sigmoid(output['logits'])
                        logits_to_compare = new_logits[batch_indices, low_range:high_range]
                        expanded_thresholds = cls_thresholds.unsqueeze(0).expand(logits_to_compare.shape)
                        target[batch_indices, low_range:high_range] = (logits_to_compare > expanded_thresholds).float().cuda(non_blocking=True)


                loss, cls_loss, kd_loss = self.compute_loss(output, 
                                                            target, 
                                                            low_range, 
                                                            old_output)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            self.scheduler.step()
            self.model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            if i % self.log_frequency == 0:
                self.logger.info(f'Epoch [{epoch + 1}/{self.nb_epochs}], Step [{str(i).zfill(3)}/{str(len(train_loader)).zfill(3)}], LR: {self.scheduler.get_last_lr()[0]:.1e}, Loss: {loss.item():.2f}, cls_loss: {cls_loss.item():.2f}, kd_loss: {kd_loss.item():.2f}')


    def compute_loss(self, output, target, low_range, old_output=None):
        """
        Input:
            1. Output from Network
            2. Ground Truth
            3. Low Range
            4. Old Output for Distillation
        Return:
            1. Total Loss
            2. Classification Loss
            3. Distillation Loss
        """
        loss = 0.
        logits = output['logits'].float()
        if old_output:
            old_logits = old_output["logits"].float()

        # Classification Loss
        cls_loss = self.cls_criterion(logits, target)

        # Distillation Loss
        kd_loss = torch.zeros(1)
        if self.kd_loss_choice and old_output:
            pod_spatial = torch.zeros(1)
            pod_flat = torch.zeros(1)
            lambda_c = self.lambda_c * math.sqrt(self.num_classes / self.task_size)
            lambda_f = self.lambda_f * math.sqrt(self.num_classes / self.task_size)
            old_features = old_output['attentions']
            new_features = output['attentions']
            pod_spatial = pod(old_features, new_features, 'spatial')
            pod_flat = embeddings_similarity(old_output['pool_embeddings'], output['pool_embeddings'])

            pod_flat = lambda_f * pod_flat
            pod_spatial = lambda_c * pod_spatial
            kd_loss = pod_flat + pod_spatial

        kd_loss = kd_loss.cuda()
        loss = cls_loss + kd_loss
        return loss, cls_loss, kd_loss


    def validate_seen(self,
                      high_range, 
                      val_loader_seen):
        self.model.eval()
        Sig = torch.nn.Sigmoid()

        preds_regular = []
        targets = []

        for image, target in val_loader_seen:
            image = image.cuda(non_blocking=True)
            target = target[:, :high_range].cuda(non_blocking=True)
            with torch.no_grad():
                with autocast():
                    output_regular = Sig(self.model(image)['logits'])
                preds_regular.append(output_regular.detach().cpu())
                targets.append(target.detach().cpu())
                torch.cuda.empty_cache()
        
        val_result = (0, 0)
        val_result2 = (0, 0, 0, 0, 0, 0)

        mAP_score_regular, score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
        mean_p_c, mean_r_c, mean_f_c, precision_o, recall_o, f1_o = calculate_metrics(torch.cat(preds_regular).cpu(), 
                                                                                      torch.cat(targets).cpu(), 
                                                                                      thre = 0.8)   
        val_result = (mAP_score_regular, score_regular)
        val_result2 = (mean_p_c, mean_r_c, mean_f_c, precision_o, recall_o, f1_o)
        return val_result, val_result2
    

    def _after_task(self, low_range, high_range, train_dataset, buffer_dataset):
        """
        Save Model and Protos
        """
        if self.replay:
            start_time = time.time()
            train_dataset.transform = self.val_transforms

            loader1 = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=self.val_batch_size,
                                                num_workers=self.num_workers)
            if low_range != 0:
                loader2 = torch.utils.data.DataLoader(buffer_dataset, 
                                                    batch_size=self.val_batch_size,
                                                    num_workers=self.num_workers)
            else:
                loader2 = None
            if self.fix_budget:
                self.num_protos = int(np.ceil(self.args.buffer_size/high_range))
                sample_ds, cls_index = sample_protos_buffer(self.model,
                                                            low_range, 
                                                            high_range,
                                                            train_dataset, 
                                                            loader1, 
                                                            loader2,
                                                            self.num_protos, 
                                                            self.threshold,
                                                            self.logger)
                self.old_dataset.add(sample_ds, cls_index)
                self.logger.info(f'Current Stage Sampled Protos: {len(sample_ds)}, Total Old Dataset Length: {len(self.old_dataset.get_bufferset())}')
                if self.label_buffer and low_range != 0 and self.old_model:
                    self.old_dataset.update(self.model,
                                            self.old_model,
                                            low_range,
                                            high_range,
                                            self.val_batch_size,
                                            self.num_workers)
                    self.logger.info(f'Old Dataset with Length: {len(self.old_dataset.get_bufferset())} are Tagged with Pseudo Label range to {high_range}')
            else:
                sample_ds = sample_protos(self.model, 
                                          low_range, 
                                          high_range,
                                          train_dataset, 
                                          loader1, 
                                          loader2, 
                                          self.num_protos, 
                                          self.threshold,
                                          self.logger)
                self.old_dataset.append(sample_ds) 
                num_all_protos = 0
                for data_sub_set in self.old_dataset:
                    num_all_protos += len(data_sub_set)
                self.logger.info(f'Current Stage Sampled Protos: {len(sample_ds)}, Total Old Dataset Length: {num_all_protos}')
                if self.label_buffer and low_range != 0 and self.old_model:
                    self.old_dataset = update_nonbuffer(self.old_dataset,
                                                        self.model,
                                                        self.old_model,
                                                        low_range,
                                                        high_range,
                                                        self.val_batch_size,
                                                        self.buffer_old_threshold,
                                                        self.buffer_new_threshold,
                                                        self.num_workers)
                    self.logger.info(f'Old Dataset with Length: {num_all_protos} are Tagged with Pseudo Label range to {high_range}')
            

            self.logger.info(f'Sample Protos Time:{time.time() - start_time}')
            self.logger.info('Saved Old Dataset')
            train_dataset.transform = self.train_transforms
        
        if self.kd_loss_choice or self.pseudo_training or self.label_buffer:
            self.old_model = deepcopy(self.model).eval()


class buffer:
    """
    Fixed Buffer
    """
    def __init__(self, buffer_size, buffer_old_threshold, buffer_new_threshold):
         assert buffer_size > 0
         self.subsets = []
         self.indexes = []
         self.buffer_size = buffer_size
         self.buffer_old_threshold = buffer_old_threshold
         self.buffer_new_threshold = buffer_new_threshold


    def add(self, subset, indexes):
        self.subsets.append(subset)
        self.indexes.append(np.stack(indexes, axis=0))
        cls_num =sum(len(sub_indexes) for sub_indexes in self.indexes)
        num_pro = int(np.ceil(self.buffer_size/cls_num))
        for i, subset in enumerate(self.subsets):
            new_index = self.indexes[i][:,:num_pro]
            self.subsets[i].indices = np.concatenate(new_index).tolist()
    
    def update(self, 
               model,
               old_model,
               low_range,
               high_range,
               val_batch_size,
               num_workers):
        buffer_task_size = len(self.subsets)
        for i, subset in enumerate(self.subsets):
            if i != (buffer_task_size - 1):
                subset_loader = build_loader(subset, val_batch_size, num_workers, phase='val')
                for j, (image, _) in enumerate(subset_loader):
                    image = image.cuda(non_blocking=True)
                    with torch.no_grad():
                        output = model(image)
                        logits = torch.sigmoid(output['logits'])
                        threshold_buffer_mask = logits[:, low_range:high_range] > self.buffer_old_threshold
                        if torch.sum(threshold_buffer_mask):
                            batch_start_idx = j * val_batch_size
                            batch_end_idx = min((j + 1) * val_batch_size, len(subset))
                            batch_indices = list(range(batch_start_idx, batch_end_idx))
                            subset.modify_items_batch(batch_indices, threshold_buffer_mask.cpu(), low_range, high_range)
            else:
                subset_loader = build_loader(subset, val_batch_size, num_workers, phase='val')
                for j, (image, _) in enumerate(subset_loader):
                    image = image.cuda(non_blocking=True)
                    with torch.no_grad():
                        output = old_model(image)
                        logits = torch.sigmoid(output['logits'])
                        threshold_buffer_mask = logits[:, :low_range] > self.buffer_new_threshold
                        if torch.sum(threshold_buffer_mask):
                            batch_start_idx = j * val_batch_size
                            batch_end_idx = min((j + 1) * val_batch_size, len(subset))
                            batch_indices = list(range(batch_start_idx, batch_end_idx))
                            subset.modify_items_batch(batch_indices, threshold_buffer_mask.cpu(), 0, low_range)

    def get_bufferset(self):
        if self.subsets:
            return torch.utils.data.ConcatDataset(self.subsets)
        else:
            return None


def update_nonbuffer(old_dataset,
                     model,
                     old_model,
                     low_range,
                     high_range,
                     val_batch_size,
                     buffer_old_threshold,
                     buffer_new_threshold,
                     num_workers):
    
    buffer_task_size = len(old_dataset)
    for i, subset in enumerate(old_dataset):
        if i != (buffer_task_size - 1):
            subset_loader = build_loader(subset, val_batch_size, num_workers, phase='val')
            for j, (image, _) in enumerate(subset_loader):
                image = image.cuda(non_blocking=True)
                with torch.no_grad():
                    output = model(image)
                    logits = torch.sigmoid(output['logits'])
                    threshold_buffer_mask = logits[:, low_range:high_range] > buffer_old_threshold
                    if torch.sum(threshold_buffer_mask):
                        batch_start_idx = j * val_batch_size
                        batch_end_idx = min((j + 1) * val_batch_size, len(subset))
                        batch_indices = list(range(batch_start_idx, batch_end_idx))
                        subset.modify_items_batch(batch_indices, threshold_buffer_mask.cpu(), low_range, high_range)
        else:
            subset_loader = build_loader(subset, val_batch_size, num_workers, phase='val')
            for j, (image, _) in enumerate(subset_loader):
                image = image.cuda(non_blocking=True)
                with torch.no_grad():
                    output = old_model(image)
                    logits = torch.sigmoid(output['logits'])
                    threshold_buffer_mask = logits[:, :low_range] > buffer_new_threshold
                    if torch.sum(threshold_buffer_mask):
                        batch_start_idx = j * val_batch_size
                        batch_end_idx = min((j + 1) * val_batch_size, len(subset))
                        batch_indices = list(range(batch_start_idx, batch_end_idx))
                        subset.modify_items_batch(batch_indices, threshold_buffer_mask.cpu(), 0, low_range)
    return old_dataset