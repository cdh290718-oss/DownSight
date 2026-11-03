"""Proposed diagnostic absolute Head joint prior in virtual camera metres."""
import json
import os
from pathlib import Path
import numpy as np
import torch


class HeadJointAnchor:
    def __init__(self, calibration_file):
        cfg = json.loads(os.environ.get('EGOSMPLX_HEAD_JOINT_ANCHOR', '{}'))
        self.enabled = cfg.get('enabled', False)
        if not isinstance(self.enabled, bool):
            raise ValueError('Head anchor enabled must be boolean')
        if not self.enabled:
            return
        if calibration_file is None:
            raise ValueError('Enabled head anchor requires explicit calibration identity')
        key = str(Path(calibration_file).resolve())
        targets = cfg['targets_by_calibration']
        if key not in targets:
            raise ValueError('No head anchor for calibration: ' + key)
        self.target = np.asarray(targets[key], dtype=np.float64)
        self.weight = float(cfg['weight'])
        self.scale_m = float(cfg['scale_m'])
        self.calibration = key
        if self.target.shape != (3,) or not np.isfinite(self.target).all():
            raise ValueError('Head anchor must be a finite XYZ vector in metres')
        if not np.isfinite([self.weight, self.scale_m]).all() or self.weight <= 0 or self.scale_m <= 0:
            raise ValueError('Head anchor weight and scale must be positive')

    def loss(self, head_joint_cam):
        if not self.enabled:
            return head_joint_cam.new_zeros(())
        if head_joint_cam.shape != (3,):
            raise ValueError('Expected native SMPL-X Head joint in camera coordinates')
        target = torch.as_tensor(self.target, dtype=head_joint_cam.dtype, device=head_joint_cam.device)
        return self.weight * ((head_joint_cam - target) / self.scale_m).square().sum()

    def report(self, head_joint_cam):
        if not self.enabled:
            return dict(enabled=False)
        if head_joint_cam is None:
            raise ValueError('Missing Head joint for anchor audit')
        value = head_joint_cam.detach().cpu().numpy()
        return dict(enabled=True, joint='SMPL-X native Head (index 15)',
                    target_head_joint_cam_m=self.target.tolist(),
                    actual_head_joint_cam_m=value.tolist(),
                    head_joint_error_m=float(np.linalg.norm(value-self.target)),
                    weight=self.weight, scale_m=self.scale_m,
                    loss=float(self.loss(head_joint_cam).detach()),
                    calibration=self.calibration,
                    supervision='user cam3 target; cam4 DFM 20-frame head mean')
