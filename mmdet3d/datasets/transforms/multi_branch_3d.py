from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms import BaseTransform
from copy import deepcopy

@TRANSFORMS.register_module()
class MultiBranch3D(BaseTransform):
    """
    Apply different augmentation branches to the same point cloud.
    Example: weak_pipeline for teacher, strong_pipeline for student.
    """

    def __init__(self, unsup_teacher=None, unsup_student=None):
        self.unsup_teacher = unsup_teacher
        self.unsup_student = unsup_student

    def transform(self, input_dict: dict) -> dict:
        # Deepcopy the input so each branch is independent
        teacher_input = deepcopy(input_dict)
        student_input = deepcopy(input_dict)

        # Apply teacher weak augmentations
        for t in self.unsup_teacher:
            teacher_input = t(teacher_input)

        # Apply student strong augmentations
        for t in self.unsup_student:
            student_input = t(student_input)

        # Combine results in the dict
        input_dict['teacher_pts'] = teacher_input['points']
        input_dict['student_pts'] = student_input['points']

        return input_dict
