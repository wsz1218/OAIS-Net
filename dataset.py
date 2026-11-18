# import os
# from torch.utils.data import DataLoader
# from torchvision import transforms as torch_transforms
#
# from pi_seg.utils.exp_imports.default import *
# from pi_seg.data.datasets.lucchi import LucchiDataset
# from pi_seg.data.datasets.skeletal_muscle import SkeletalMuscleDataset
# from pi_seg.data.transforms import *
#
# from config import MIN_OBJECT_AREA, DATASET_NAME, DATASET_DIR, BATCH_SIZE, VAL_BATCH_SIZE, NUM_MAX_POINTS
#
#
# # points sampler
# points_sampler = MultiPointSampler(NUM_MAX_POINTS, prob_gamma=0.80,
#                                    merge_objects_prob=0.15,
#                                    max_num_merged_objects=2,
#                                    use_hierarchy=False,
#                                    first_click_center=False)
#
# # Define data transformations
# transform_train = torch_transforms.Compose([ToTensor(), Resize(), RandomCrop(), RandomFlip(), Rotate()])
# transform_val = torch_transforms.Compose([ToTensor(), Resize()])
#
# # Select dataset dynamically
# if DATASET_NAME == "Lucchi":
#     DatasetClass = LucchiDataset
# elif DATASET_NAME == "SkeletalMuscle":
#     DatasetClass = SkeletalMuscleDataset
# else:
#     raise ValueError("Invalid dataset name! Choose between 'Lucchi' and 'SkeletalMuscle'.")
#
# # Training dataset
# trainset = DatasetClass(
#     dataset_path=os.path.join(DATASET_DIR, 'train'),
#     images_dir_name='input',
#     masks_dir_name='target',
#     augmentator=transform_train,
#     min_object_area=MIN_OBJECT_AREA,
#     keep_background_prob=0.05,
#     points_sampler=points_sampler,
# )
#
# # Validation dataset
# valset = DatasetClass(
#     dataset_path=os.path.join(DATASET_DIR, 'test'),
#     images_dir_name='input',
#     masks_dir_name='target',
#     augmentator=transform_val,
#     min_object_area=MIN_OBJECT_AREA,
#     points_sampler=points_sampler,
# )
#
# # DataLoaders
# train_dataloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8)
# val_dataloader = DataLoader(valset, batch_size=VAL_BATCH_SIZE, pin_memory=True, num_workers=8)
#
#
# # --- Display Dataset Information ---
# print("\n" + "="*50)
# print("DATASET & DATALOADER INFORMATION")
# print("="*50)
# print(f"[*] Dataset Name : {DATASET_NAME}")
# print(f"[*] Dataset Path : {DATASET_DIR}")
# # -----------------------------------
#
#
# # --- Display Dataloader Information ---
# print(f"[*] Train Samples      : {len(trainset)} images, {len(train_dataloader)} batches")
# print(f"[*] Train Batch Size   : {BATCH_SIZE}")
# print(f"[*] Validation Samples : {len(valset)} images, {len(val_dataloader)} batches")
# print(f"[*] Validation Batch Size: {VAL_BATCH_SIZE}")
# print("="*50 + "\n")
# # ------------------------------------

import os
import random
import math
from collections import defaultdict
from torch.utils.data import DataLoader, ConcatDataset, Sampler
from torchvision import transforms as torch_transforms

from pi_seg.utils.exp_imports.default import *
from pi_seg.data.datasets.skeletal_muscle import SkeletalMuscleDataset
from pi_seg.data.transforms import *

from config import MIN_OBJECT_AREA, DATASET_DIR, BATCH_SIZE, VAL_BATCH_SIZE, NUM_MAX_POINTS


# --- 1. 同源 Batch 采样器定义 ---
class DomainHomogeneousBatchSampler(Sampler):
    def __init__(self, concat_dataset: ConcatDataset, batch_size: int, drop_last: bool = True):
        super().__init__(concat_dataset)
        self.concat_dataset = concat_dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.domain_to_indices = defaultdict(list)
        current_idx = 0
        for domain_id, dataset in enumerate(self.concat_dataset.datasets):
            dataset_len = len(dataset)
            self.domain_to_indices[domain_id] = list(range(current_idx, current_idx + dataset_len))
            current_idx += dataset_len

    def __iter__(self):
        domain_pools = {k: list(v) for k, v in self.domain_to_indices.items()}
        for k in domain_pools:
            random.shuffle(domain_pools[k])

        available_domains = list(self.domain_to_indices.keys())

        while available_domains:
            chosen_domain = random.choice(available_domains)
            pool = domain_pools[chosen_domain]

            if len(pool) < self.batch_size:
                if not self.drop_last and len(pool) > 0:
                    batch_indices = pool[:]
                    domain_pools[chosen_domain] = []
                    yield batch_indices
                available_domains.remove(chosen_domain)
                continue

            batch_indices = pool[:self.batch_size]
            domain_pools[chosen_domain] = pool[self.batch_size:]
            yield batch_indices

    def __len__(self):
        if self.drop_last:
            return sum(len(indices) // self.batch_size for indices in self.domain_to_indices.values())
        else:
            return sum(math.ceil(len(indices) / self.batch_size) for indices in self.domain_to_indices.values())


# --- 2. 基础配置与转换 ---
points_sampler = MultiPointSampler(NUM_MAX_POINTS, prob_gamma=0.80,
                                   merge_objects_prob=0.15,
                                   max_num_merged_objects=2,
                                   use_hierarchy=False,
                                   first_click_center=False)

transform_train = torch_transforms.Compose([ToTensor(), Resize(), RandomCrop(), RandomFlip(), Rotate()])
transform_val = torch_transforms.Compose([ToTensor(), Resize()])

# --- 3. 动态扫描目录并组装多域数据集 ---
train_datasets = []
val_datasets = []

# 扫描 DATASET_DIR 下的 domain 文件夹。
# CLICK_ENCODING_DOMAIN 可用于点击编码消融的单域实验，例如：domain_2_cells。
selected_domain = os.environ.get("CLICK_ENCODING_DOMAIN")
all_domain_folders = [
    f for f in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, f))
]
if selected_domain:
    if selected_domain not in all_domain_folders:
        raise ValueError(
            f"CLICK_ENCODING_DOMAIN={selected_domain} not found under {DATASET_DIR}. "
            f"Available domains: {all_domain_folders}"
        )
    domain_folders = [selected_domain]
else:
    domain_folders = all_domain_folders

for domain_name in domain_folders:
    domain_path = os.path.join(DATASET_DIR, domain_name)

    # 构建训练子集
    train_dir = os.path.join(domain_path, 'train')
    if os.path.exists(train_dir):
        train_datasets.append(SkeletalMuscleDataset(
            dataset_path=train_dir,
            images_dir_name='input',
            masks_dir_name='target',
            augmentator=transform_train,
            min_object_area=MIN_OBJECT_AREA,
            keep_background_prob=0.05,
            points_sampler=points_sampler,
        ))

    # 构建验证子集
    val_dir = os.path.join(domain_path, 'test')
    if os.path.exists(val_dir):
        val_datasets.append(SkeletalMuscleDataset(
            dataset_path=val_dir,
            images_dir_name='input',
            masks_dir_name='target',
            augmentator=transform_val,
            min_object_area=MIN_OBJECT_AREA,
            points_sampler=points_sampler,
        ))

combined_trainset = ConcatDataset(train_datasets)
combined_valset = ConcatDataset(val_datasets)

# --- 4. 构建 DataLoaders ---
# 训练集使用同源采样器
train_sampler = DomainHomogeneousBatchSampler(combined_trainset, batch_size=BATCH_SIZE, drop_last=True)
train_dataloader = DataLoader(combined_trainset, batch_sampler=train_sampler, pin_memory=True, num_workers=8)

# 验证集不需要同源采样，正常顺序加载即可
val_dataloader = DataLoader(combined_valset, batch_size=VAL_BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8)

# --- 5. 信息打印 ---
print("\n" + "=" * 50)
print("MULTI-DOMAIN DATALOADER INFORMATION")
print("=" * 50)
print(f"[*] Base Dataset Path : {DATASET_DIR}")
print(f"[*] Detected Domains  : {len(domain_folders)} ({', '.join(domain_folders[:3])}...)")
print(f"[*] Train Samples     : {len(combined_trainset)} images, {len(train_dataloader)} pure batches")
print(f"[*] Train Batch Size  : {BATCH_SIZE} (Homogeneous)")
print(f"[*] Val Samples       : {len(combined_valset)} images, {len(val_dataloader)} batches")
print(f"[*] Val Batch Size    : {VAL_BATCH_SIZE}")
print("=" * 50 + "\n")
