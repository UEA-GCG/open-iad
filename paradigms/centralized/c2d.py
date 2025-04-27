import yaml
import time

from configuration.device import assign_service
from configuration.registration import setting_name, dataset_name, model_name
from data_io.data_holder import DataHolder
from tools.utils import *
from rich import print
from augmentation.type import aug_type
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

class CentralizedAD2D():
    def __init__(self, args):
        self.args = args

    def load_config(self):
        with open('./configuration/3_dataset_base/{}.yaml'.format(self.args.dataset), 'r', encoding='utf-8') as f:
            config_dataset = yaml.load(f, Loader=yaml.SafeLoader)
        with open('./configuration/2_train_base/centralized_learning.yaml', 'r', encoding='utf-8') as f:
            config_train = yaml.load(f, Loader=yaml.SafeLoader)
        with open('./configuration/1_model_base/{}.yaml'.format(self.args.model), 'r', encoding='utf-8') as f:
            config_model = yaml.load(f, Loader=yaml.SafeLoader)
        
        config = override_config(config_dataset, config_train)
        config = override_config(config, config_model)
        self.para_dict = merge_config(config, self.args)  # 把config文件中参数融合入yaml文件中
        self.args = extract_config(self.args)

        # ip, root_path = assign_service(self.para_dict['server_moda'])
        ip = '127.0.0.1'
        root_path = self.para_dict['root_path']
        print('local ip: {}, root_path: {}'.format(ip, root_path))

        self.para_dict['root_path'] = root_path
        self.para_dict['data_path'] = '{}{}'.format(root_path, self.para_dict['data_path'])
        self.para_dict['file_path'] = record_path(self.para_dict)  # 创建存档路径

        if self.para_dict['save_log']:
            save_arg(self.para_dict, self.para_dict['file_path'])
            save_script(__file__, self.para_dict['file_path'])

        self.print_info()  # 打印config中非空参数和所有参数
        self.check_args()  # 判断是否只有1个学习范式

    def check_args(self):
        n = 0
        for s in setting_name:
            n += self.para_dict[s]

        if n == 0:
            raise ValueError('Please Assign Learning Paradigm!')
        if n >= 2:
            raise ValueError('There Are Multiple Flags of Paradigm!')

    def print_info(self):
        print('----------args-----------')
        print(self.args)
        print('----------para_dict-----------')
        print(self.para_dict)
        print('---------------------')

    def load_data(self):
        dataset_package = __import__(dataset_name[self.para_dict['dataset']][0])
        dataset_module = getattr(dataset_package, dataset_name[self.para_dict['dataset']][1])
        dataset_class = getattr(dataset_module, dataset_name[self.para_dict['dataset']][2])

        dataloader = DataHolder(dataset_class, self.para_dict)  # dataset.small_tool.Small_Tool
        dataloader.create()

        self.chosen_train_loaders = dataloader.chosen_train_loaders
        self.chosen_valid_loaders = dataloader.chosen_valid_loaders
        self.chosen_vis_loaders = dataloader.chosen_vis_loaders

        # vanilla 模式时以下为空
        self.chosen_transfer_train_loaders = dataloader.chosen_transfer_train_loaders
        self.chosen_transfer_valid_loaders = dataloader.chosen_transfer_valid_loaders
        self.chosen_transfer_vis_loaders = dataloader.chosen_transfer_vis_loaders

        self.class_name = dataloader.class_name
    
    def init_model(self):
        model_package = __import__(model_name[self.para_dict['model']][0])
        model_module = getattr(model_package, model_name[self.para_dict['model']][1])
        model_class = getattr(model_module, model_name[self.para_dict['model']][2])
        self.trainer = model_class(self.para_dict)  # arch.patchcore.PatchCore
    
    def train_and_infer(self, train_loaders, valid_loaders, vis_loaders, train_task_ids, valid_task_ids):
        # train all task in one time
        for i, train_loader in enumerate(train_loaders):
            print('-> train ...')
            self.para_dict['train_task_id_tmp'] = train_task_ids[i]
            print('run task: {}, {}'.format(self.para_dict['train_task_id_tmp'], self.class_name[self.para_dict['train_task_id_tmp']]))
            self.trainer.train_model(train_loader, i)  # arch.patchcore.PatchCore

            print('-> test ...')
            # test each task individually
            for j, (valid_loader, vis_loader) in enumerate(zip(valid_loaders, vis_loaders)):
                # for continual
                if j > i:
                    break
                self.para_dict['valid_task_id_tmp'] = valid_task_ids[j]
                print('run task: {}, {}'.format(self.para_dict['valid_task_id_tmp'], self.class_name[self.para_dict['valid_task_id_tmp']]))
                
                # calculate time 
                start_time = time.time()
                self.trainer.prediction(valid_loader, j)
                end_time = time.time()
                inference_speed = (end_time - start_time)/len(self.trainer.img_path_list)

                # calculate result
                pixel_auroc, img_auroc, pixel_ap, img_ap, pixel_aupro = self.trainer.cal_metric_all(
                    task_id=int(self.para_dict['train_task_id_tmp']))
                self.trainer.recorder.update(self.para_dict)  # 更新参数配置 recorder在base.py内

                paradim = self.trainer.recorder.paradigm_name()
                infor_basic = 'paradigm: {}, dataset: {}, model: {}, train_task_id: {}, valid_task_id: {}'.format(paradim,
                    self.para_dict['dataset'], self.para_dict['model'], self.para_dict['train_task_id_tmp'], self.para_dict['valid_task_id_tmp'])
                infor_result = 'pixel_auroc: {:.4f}, img_auroc: {:.4f}, pixel_ap: {:.4f}, img_ap: {:.4f}, pixel_aupro: {:.4f}, inference speed: {:.4f}'.format(
                    pixel_auroc, img_auroc, pixel_ap, img_ap, pixel_aupro, inference_speed)
                self.trainer.recorder.printer('{} {}'.format(infor_basic, infor_result))

                # save result
                if self.para_dict['save_log']:
                    self.trainer.recorder.record_result(infor_result)
                    self.trainer.record_detail_result()  # GUO:写详细的得分情况在result文件

                # 可视化 保存可视化图像 这里会把前面prediction预测的结果清零
                if self.para_dict['vis']:
                    print('-> visualize ...')
                    self.trainer.visualization(vis_loader, j)

    def work_flow(self):
        if self.para_dict['vanilla'] or self.para_dict['semi'] or self.para_dict['fewshot'] or self.para_dict['noisy'] or self.para_dict['continual']:
            self.train_and_infer(self.chosen_train_loaders, self.chosen_valid_loaders, self.chosen_vis_loaders,
                                  self.para_dict['train_task_id'], self.para_dict['valid_task_id'])
        if self.para_dict['transfer']:
            self.train_and_infer(self.chosen_train_loaders, self.chosen_valid_loaders, self.chosen_vis_loaders,
                                  self.para_dict['train_task_id'], self.para_dict['train_task_id'])
            self.train_and_infer(self.chosen_transfer_train_loaders, self.chosen_transfer_valid_loaders, self.chosen_transfer_vis_loaders,
                                  self.para_dict['valid_task_id'], self.para_dict['valid_task_id'])

    # 所有任务重新从文件夹中读硬盘和训练
    def train_small_tool(self):
        self.load_config()
        self.load_data()
        self.init_model()

        # 遍历所有task，实际也可以只指定一个task id，这样就是每个类单独训练
        for i, train_loader in enumerate(self.chosen_train_loaders):
            print('-> train ...')
            self.para_dict['train_task_id_tmp'] = self.para_dict['train_task_id'][i]
            print('run task: {}, {}'.format(self.para_dict['train_task_id_tmp'],
                                            self.class_name[self.para_dict['train_task_id_tmp']]))
            self.trainer.train_model(train_loader, i, self.class_name[self.para_dict['train_task_id_tmp']])

    # 初始化模型，加载硬盘中处理好的数据
    def init_small_tool(self, task_id):
        self.load_config()

        dataset_package = __import__(dataset_name[self.para_dict['dataset']][0])
        dataset_module = getattr(dataset_package, dataset_name[self.para_dict['dataset']][1])
        dataset_class = getattr(dataset_module, dataset_name[self.para_dict['dataset']][2])
        name_class = getattr(dataset_module, 'small_tool_classes')
        self.class_name = name_class()
        self.dataset_class = dataset_class

        self.init_model()
        self.para_dict['train_task_id_tmp'] = self.para_dict['train_task_id'][task_id]
        print('init task: {}, {}'.format(self.para_dict['train_task_id_tmp'],
                                        self.class_name[self.para_dict['train_task_id_tmp']]))
        self.trainer.load_model(self.class_name[self.para_dict['train_task_id_tmp']])

    # 单张图片、预测阈值指标
    def predict_small_tool(self, img, device_type, conf=0.5):
        pass



    # 批量训练和批量测试
    def run_work_flow(self):
        self.load_config()
        self.load_data()
        self.init_model()
        self.work_flow()
