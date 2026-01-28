from mmengine.dataset import BaseDataset
from mmdet3d.registry import DATASETS
from mmengine.registry import FUNCTIONS
from mmcv.transforms import Compose
from torch.utils.data.dataloader import default_collate

@DATASETS.register_module()
class MTCombinedDataset:
    """
    Mean-Teacher Combined Dataset for semi-supervised 3D detection.
    
    Returns paired samples from labeled (source) and unlabeled (target) datasets.
    Unlabeled samples use two different augmentation pipelines (weak/strong).
    
    Output format per sample:
    {
        'labeled': dict(inputs=..., data_samples=...),
        'unlabeled': dict(weak=..., strong=...)
    }
    """
    
    def __init__(self,
                 labeled_dataset,
                 unlabeled_weak_dataset,
                 unlabeled_strong_dataset,
                 **kwargs):

        # Build sub-datasets
        self.labeled_dataset = DATASETS.build(labeled_dataset)
        self.unlabeled_weak_dataset = DATASETS.build(unlabeled_weak_dataset)
        self.unlabeled_strong_dataset = DATASETS.build(unlabeled_strong_dataset)

        self.labeled_len = len(self.labeled_dataset)
        self.unlabeled_len = len(self.unlabeled_weak_dataset)
        
        # Sanity check
        assert len(self.unlabeled_weak_dataset) == len(self.unlabeled_strong_dataset), \
            "Weak and strong unlabeled datasets must have the same length"

    def __len__(self):
        # Return max to ensure we use all data
        return max(self.labeled_len, self.unlabeled_len)

    def __getitem__(self, idx):
        """
        Returns a dict with labeled and unlabeled data.
        """
        labeled = self.labeled_dataset[idx % self.labeled_len]
        unlabeled_weak = self.unlabeled_weak_dataset[idx % self.unlabeled_len]
        unlabeled_strong = self.unlabeled_strong_dataset[idx % self.unlabeled_len]

        return {
            "labeled": labeled,
            "unlabeled": {
                "weak": unlabeled_weak,
                "strong": unlabeled_strong
            }
        }

@FUNCTIONS.register_module()
def mean_teacher_collate_fn(data_batch):
    """
    Custom collate function for Mean-Teacher dataset.

    Converts list of dicts into the format expected by MeanTeacher3DDetector:
    
    Input (from dataloader):
        List of samples, each: {
            'labeled': {'inputs': ..., 'data_samples': ...},
            'unlabeled': {'weak': {...}, 'strong': {...}
            }
        }
    
    Output:
        batch_inputs_dict = {
            'labeled': batched_inputs,
            'unlabeled': {
                'weak': batched_inputs,
                'strong': batched_inputs
            }
        }
        batch_data_samples = {
            'labeled': [data_sample_1, ...],
            'unlabeled': [data_sample_1, ...]
        }
    """
    # Separate labeled and unlabeled data
    labeled_samples = [sample['labeled'] for sample in data_batch]
    unlabeled_weak_samples = [sample['unlabeled']['weak'] for sample in data_batch]
    unlabeled_strong_samples = [sample['unlabeled']['strong'] for sample in data_batch]
    
    # Helper function to collate inputs and data_samples
    def collate_samples(samples):
        """
        Collate samples with variable-sized point clouds.
        Keep points as a list, but batch other tensor data.
        """
        # Separate inputs and data_samples
        inputs_list = [s['inputs'] for s in samples]
        data_samples_list = [s['data_samples'] for s in samples]
        
        # Keep points as list (not stacked) to avoid size mismatch
        batch_inputs = {
            'points': [inp['points'] for inp in inputs_list]
        }
        
        return batch_inputs, data_samples_list
    
    # Collate each component
    labeled_inputs, labeled_data_samples = collate_samples(labeled_samples)
    weak_inputs, weak_data_samples = collate_samples(unlabeled_weak_samples)
    strong_inputs, strong_data_samples = collate_samples(unlabeled_strong_samples)
    
    # Structure output as expected by MeanTeacher3DDetector
    batch_inputs_dict = {
        'labeled': labeled_inputs,
        'unlabeled': {
            'weak': weak_inputs,
            'strong': strong_inputs
        }
    }
    
    batch_data_samples = {
        'labeled': labeled_data_samples,
        'unlabeled': {
            'weak': weak_data_samples,
            'strong': strong_data_samples
        }
    }
    
    return batch_inputs_dict, batch_data_samples