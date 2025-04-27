from torchvision import transforms as T
from augmentation.cutpaste_aug import *

__all__ = ['aug_type']

def aug_type(augment_type, args):
    if augment_type == 'normal':
        img_transform = T.Compose([T.Resize((args['data_size'], args['data_size'])),
                                    T.CenterCrop(args['data_crop_size']),
                                    T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

        mask_transform = T.Compose([T.Resize(args['mask_size']),
                                    T.CenterCrop(args['mask_crop_size']),
                                    T.ToTensor(),
                                    ])
    
    elif augment_type == 'cutpaste':
        after_cutpaste_transform = T.Compose([T.RandomRotation(90),
                                              T.ToTensor(),
                                              T.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
                                            ])
        
        img_transform = T.Compose([T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                   T.Resize((args['data_crop_size'], args['data_crop_size'])),
                                   CutPasteNormal(transform=after_cutpaste_transform)
                                   #T.RandomChoice([CutPasteNormal(transform=after_cutpaste_transform),
                                   #                CutPasteScar(transform=after_cutpaste_transform)])
                                   ])

        mask_transform = T.Compose([T.Resize(args['mask_size']),
                                    T.CenterCrop(args['mask_crop_size']),
                                    T.ToTensor(),
                                    ])

    elif augment_type == 'small_tool':  # 小工具预处理，不需要crop
        img_transform = T.Compose([
            # T.Lambda(lambda img: padding_to_square(img)),  # TODO:是否需要填充操作待验证
            T.Resize((args['data_size'], args['data_size'])),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        mask_transform = T.Compose([
        #     # T.Lambda(lambda img: padding_to_square(img)),  # 填充操作
            T.Resize(args['mask_size']),
            T.ToTensor(),
            ])
    else:
        raise NotImplementedError('The Augmentation Type Has Not Been Implemented Yet')

    return img_transform, mask_transform

from torchvision.transforms import functional as F
# 不改变图像比例
def padding_to_square(img):
    width, height = img.size  # 获取图像的原始宽度和高度
    max_size = max(width, height)  # 选择较长的边作为正方形的边长

    # 计算需要填充的宽度和高度
    pad_left = (max_size - width) // 2
    pad_right = max_size - width - pad_left
    pad_top = (max_size - height) // 2
    pad_bottom = max_size - height - pad_top

    # 填充图像
    return F.pad(img,[pad_left, pad_top, pad_right, pad_bottom], fill=0)