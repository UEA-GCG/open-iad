import os
import cv2
from tools.visualize import save_anomaly_map
from configuration.registration import setting_name
from rich import print
from sklearn.metrics import roc_curve
import numpy as np

__all__ = ['RecordHelper']

class RecordHelper():
    def __init__(self, config):
        self.config = config

    def update(self, config):
        self.config = config
    
    def printer(self, info):
        print(info)

    def paradigm_name(self):
        for s in setting_name:
            if self.config[s]:
                return s

        print('Add new setting in record_helper.py!')
        return 'unknown'

    def record_result(self, result):
        paradim = self.paradigm_name()
        save_dir = '{}/benchmark/{}/{}/{}/task_{}'.format(self.config['work_dir'], paradim, self.config['dataset'],
                                                     self.config['model'], self.config['train_task_id_tmp'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = save_dir + '/result.txt'
        if paradim == 'vanilla':
            save_path = save_path
        if paradim == 'semi':
            save_path = '{}/result_{}_num.txt'.format(save_dir, self.config['semi_anomaly_num'])
        if paradim == 'fewshot':
            save_path = '{}/result_{}_{}_shot.txt'.format(save_dir, ''.join(self.config['fewshot_aug_type']), self.config['fewshot_exm'])
        if paradim == 'continual':
            save_path = '{}/result_{}_task.txt'.format(save_dir, self.config['valid_task_id_tmp'])
        if paradim == 'noisy':
            save_path = '{}/result_{}_ratio.txt'.format(save_dir, self.config['noisy_ratio'])
        if paradim == 'transfer':
            save_path = '{}/result_from_{}_to_{}.txt'.format(save_dir, self.config['train_task_id'][0], self.config['valid_task_id'][0]) 

        with open(save_path, 'w') as f: # 每次重新写
            print(result, file=f) 

    def record_images(self, img_pred_list, img_gt_list, pixel_pred_list, pixel_gt_list, img_path_list):
        paradim = self.paradigm_name()
        save_dir = '{}/benchmark/{}/{}/{}/task_{}'.format(self.config['work_dir'], paradim, self.config['dataset'],
                                                      self.config['model'], self.config['train_task_id_tmp'])
        
        if paradim == 'vanilla':
            save_dir = save_dir + '/vis'
        if paradim == 'semi':
            save_dir = '{}/vis_{}_num'.format(save_dir, self.config['semi_anomaly_num'])
        if paradim == 'fewshot':
            save_dir = '{}/vis_{}_{}_shot'.format(save_dir, ''.join(self.config['fewshot_aug_type']), self.config['fewshot_exm'])
        if paradim == 'continual':
            save_dir = '{}/vis_{}_task'.format(save_dir, self.config['valid_task_id_tmp'])
        if paradim == 'noisy':
            save_dir = '{}/vis_{}_ratio'.format(save_dir, self.config['noisy_ratio'])
        if paradim == 'transfer':
            save_dir = '{}/vis_from_{}_to_{}'.format(save_dir, self.config['train_task_id'][0], self.config['valid_task_id'][0])

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        for i in range(len(img_path_list)):
            img_src = cv2.imread(img_path_list[i][0])
            img_src = cv2.resize(img_src, pixel_pred_list[0].shape)
            path_dir = img_path_list[i][0].split('\\')
            save_path = '{}/{}_{}'.format(save_dir, path_dir[-2], path_dir[-1][:-4])

            save_anomaly_map(anomaly_map=pixel_pred_list[i], input_img=img_src, mask=pixel_gt_list[i], file_path=save_path)

    def record_detail(self, img_pred_list, img_gt_list, img_path_list):
        paradim = self.paradigm_name()
        save_dir = '{}/benchmark/{}/{}/{}/task_{}'.format(
            self.config['work_dir'],
            paradim,
            self.config['dataset'],
            self.config['model'],
            self.config['train_task_id_tmp'])

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, 'result.txt')

        # roc curve and best threshold
        fpr, tpr, thresholds = roc_curve(img_gt_list, img_pred_list)
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        optimal_tpr = tpr[optimal_idx]

        pred_binary_list = [1 if pred >= optimal_threshold else 0 for pred in img_pred_list]

        correct = sum(1 for pred, gt in zip(pred_binary_list, img_gt_list) if pred==gt)
        total = len(img_gt_list)
        acc = correct/total if total > 0 else 0.0

        TP = sum(1 for pred, gt in zip(pred_binary_list, img_gt_list) if pred == 1 and gt == 1)
        # TN = sum(1 for pred, gt in zip(pred_binary_list, img_gt_list) if pred == 0 and gt == 0)
        FP = sum(1 for pred, gt in zip(pred_binary_list, img_gt_list) if pred == 1 and gt == 0)
        # FN = sum(1 for pred, gt in zip(pred_binary_list, img_gt_list) if pred == 0 and gt == 1)

        # jing que lv
        Precision = TP / (TP+FP) if (TP+FP)>0 else 0.0

        try:
            with open(save_path, 'a', encoding='utf-8') as f:
                f.write(f"Optimal Threshold:{optimal_threshold:.3f}\n")
                f.write(f"Optimal TPR:{optimal_tpr:.3f}, Optimal FPR:{optimal_fpr:.3f}\n")
                f.write(f"Accuracy:{acc:.3f}\n")
                f.write(f"Precision:{Precision:.3f}\n")
                f.write("Prediction\tPredcition(Threshold)\tGround Truth\tImage Path\n")
                # 逐行写入每个样本的信息
                for img_path, pred, pred_binary, gt in zip(img_path_list, img_pred_list, pred_binary_list, img_gt_list):
                    f.write(f"{pred:.3f}\t{pred_binary}\t{gt}\t{img_path}\n")

        except Exception as e:
            print(f"Error writing to {save_path}: {e}")
            raise
