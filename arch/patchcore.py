import numpy as np
from arch.base import ModelBase
from models.patchcore.patchcore import PatchCore as patchcore_official
from models.patchcore import common
from models.patchcore import sampler
from torchvision import models

__all__ = ['PatchCore']

class PatchCore(ModelBase):
    def __init__(self, config):
        super(PatchCore, self).__init__(config)
        self.config = config

        # TODO: 这里更改骨干网
        if self.config['net'] == 'resnet18': 
            self.net = models.resnet18(pretrained=True, progress=True).to(self.device)
        if self.config['net'] == 'wide_resnet50':
            self.net = models.wide_resnet50_2(pretrained=True, progress=True).to(self.device)

        self.sampler = self.get_sampler(self.config['_sampler_name'])
        self.nn_method = common.FaissNN(self.config['_faiss_on_gpu'], self.config['_faiss_num_workers'])

        self.patchcore_instance = patchcore_official(self.device)
        self.patchcore_instance.load(
            backbone=self.net,
            layers_to_extract_from=self.config['_layers_to_extract_from'],  # ['layer2', 'layer3']
            device=self.device,  # device(type='cuda', index=0)
            input_shape=self.config['_input_shape'],  # [3, 256, 256]
            pretrain_embed_dimension=self.config['_pretrain_embed_dimension'],  # 1024
            target_embed_dimension=self.config['_target_embed_dimension'],  # 1024
            patchsize=self.config['_patch_size'],  # 3
            featuresampler=self.sampler,  # approx_greedy_coreset
            anomaly_scorer_num_nn=self.config['_anomaly_scorer_num_nn'],  # 1
            nn_method=self.nn_method,  # faiss
        )

    def get_sampler(self, name):
        if name == 'identity':
            return sampler.IdentitySampler()
        elif name == 'greedy_coreset':
            return sampler.GreedyCoresetSampler(self.config['sampler_percentage'], self.device)
        elif name == 'approx_greedy_coreset':
            return sampler.ApproximateGreedyCoresetSampler(self.config['sampler_percentage'], self.device)
        else:
            raise ValueError('No This Sampler: {}'.format(name))

    def train_model(self, train_loader, task_id, inf=''):
        self.patchcore_instance.eval()
        self.patchcore_instance.fit(train_loader)

        # 保存数据
        # self.patchcore_instance.save_data('{}/{}/{}/{}_{}'.format(self.config['work_dir'],
        #                                                     self.config['learning_mode'],
        #                                                     self.config['dataset'],
        #                                                     inf,'data.npy'))

    # def load_model(self, inf=''):
    #     self.patchcore_instance.eval()
        #加载数据
        # self.patchcore_instance.load_data('{}/{}/{}/{}_{}'.format(self.config['work_dir'],
        #                                                     self.config['learning_mode'],
        #                                                     self.config['dataset'],
        #                                                     inf, 'data.npy'))
        # self.patchcore_instance.init_fit()


    def prediction(self, valid_loader, task_id=None):
        self.patchcore_instance.eval()
        self.clear_all_list()

        scores, segmentations, labels_gt, masks_gt, img_srcs = self.patchcore_instance.predict(valid_loader)

        scores = np.array(scores)
        min_scores = scores.min(axis=-1).reshape(-1, 1)
        max_scores = scores.max(axis=-1).reshape(-1, 1)
        scores = (scores - min_scores) / (max_scores - min_scores)
        scores = np.mean(scores, axis=0)

        segmentations = np.array(segmentations)

        # 原代码
        # min_scores = segmentations.reshape(len(segmentations), -1).min(axis=-1).reshape(-1, 1, 1, 1)
        # max_scores = segmentations.reshape(len(segmentations), -1).max(axis=-1).reshape(-1, 1, 1, 1)
        # segmentations = (segmentations - min_scores) / (max_scores - min_scores)
        # segmentations = np.mean(segmentations, axis=0)  # ??

        # 改后代码
        min_scores = segmentations.reshape(len(segmentations), -1).min(axis=-1).reshape(-1, 1, 1)
        max_scores = segmentations.reshape(len(segmentations), -1).max(axis=-1).reshape(-1, 1, 1)
        segmentations = (segmentations - min_scores) / (max_scores - min_scores)
        segmentations[segmentations >= 0.5] = 1
        segmentations[segmentations < 0.5] = 0

        segmentations = np.array(segmentations, dtype='uint8')
        masks_gt = np.array(masks_gt).squeeze().astype(int)

        self.pixel_gt_list = [mask for mask in masks_gt]
        self.pixel_pred_list = [seg for seg in segmentations]
        self.img_gt_list = labels_gt
        self.img_pred_list = scores
        self.img_path_list = img_srcs