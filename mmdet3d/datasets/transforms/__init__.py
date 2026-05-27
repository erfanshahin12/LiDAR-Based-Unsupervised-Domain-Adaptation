# Copyright (c) OpenMMLab. All rights reserved.
from .dbsampler import DataBaseSampler
from .formating import Pack3DDetInputs
from .loading import (LidarDet3DInferencerLoader, LoadAnnotations3D,
                      LoadImageFromFileMono3D, LoadMultiViewImageFromFiles,
                      LoadPointsFromDict, LoadPointsFromFile,
                      LoadPointsFromMultiSweeps, MonoDet3DInferencerLoader,
                      MultiModalityDet3DInferencerLoader, NormalizePointsColor,
                      PointSegClassMapping)
from .test_time_aug import MultiScaleFlipAug3D
# yapf: disable
from .transforms_3d import (AffineResize, BackgroundPointsFilter,
                            GlobalAlignment, GlobalRotScaleTrans,
                            IndoorPatchPointSample, IndoorPointSample,
                            LaserMix, MultiViewWrapper, ObjectNameFilter,
                            ObjectNoise, ObjectRangeFilter, ObjectSample,
                            PhotoMetricDistortion3D, PointSample, PointShuffle,
                            PointsRangeFilter, PolarMix, RandomDropPointsColor,
                            RandomFlip3D, RandomJitterPoints, RandomResize3D,
                            RandomShiftScale, Resize3D, VoxelBasedPointSampler)

#from .load_empty_ann3d import LoadEmptyAnnotations3D
#from .multi_branch_3d import MultiBranch3D
from .transform_KittiToNus import KittiToNuscenes
from .transform_NusToKitti import NuscenesToKitti
from .class_remap import (ClassRemap, ClassRemapWithLabel)
from .hard_instance_mining import (HardInstanceBank, HardInstanceSampling, build_hard_instance_bank)
from .random_object_scaling import RandomObjectScaling

__all__ = [
    'ObjectSample', 'RandomFlip3D', 'ObjectNoise', 'GlobalRotScaleTrans',
    'PointShuffle', 'ObjectRangeFilter', 'PointsRangeFilter',
    'Pack3DDetInputs', 'LoadMultiViewImageFromFiles', 'LoadPointsFromFile',
    'DataBaseSampler', 'NormalizePointsColor', 'LoadAnnotations3D',
    'IndoorPointSample', 'PointSample', 'PointSegClassMapping',
    'MultiScaleFlipAug3D', 'LoadPointsFromMultiSweeps',
    'BackgroundPointsFilter', 'VoxelBasedPointSampler', 'GlobalAlignment',
    'IndoorPatchPointSample', 'LoadImageFromFileMono3D', 'ObjectNameFilter',
    'RandomDropPointsColor', 'RandomJitterPoints', 'AffineResize',
    'RandomShiftScale', 'LoadPointsFromDict', 'Resize3D', 'RandomResize3D',
    'MultiViewWrapper', 'PhotoMetricDistortion3D', 'MonoDet3DInferencerLoader',
    'LidarDet3DInferencerLoader', 'PolarMix', 'LaserMix',
    'MultiModalityDet3DInferencerLoader',
    #'LoadEmptyAnnotations3D', 'MultiBranch3D',                                     # custom made
    'KittiToNuscenes', 'NuscenesToKitti', 'ClassRemap', 'ClassRemapWithLabel',      # custom made
    'HardInstanceBank', 'HardInstanceSampling', 'build_hard_instance_bank',         # custom made
    'RandomObjectScaling',                                                           # custom made
]
