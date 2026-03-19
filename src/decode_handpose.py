# Copyright 2026, POV Label Tracker
# Custom YOLO pose decoder for 21 hand keypoints (instead of 17 COCO body keypoints)
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Union

import numpy as np
from typing_extensions import Self

from axelera import types
from axelera.app import logging_utils
from axelera.app.meta.base import MetaObject
from axelera.app.meta.keypoint import (
    BottomUpKeypointDetectionMeta,
    KeypointObjectWithBbox,
)
from axelera.app.meta.gst_decode_utils import decode_bbox
from axelera.app.operators import PipelineContext
from ax_models.decoders.yolopose import DecodeYoloPose

LOG = logging_utils.getLogger(__name__)


@dataclass(frozen=True)
class HandKeypointsMeta(BottomUpKeypointDetectionMeta):
    """Metadata for 21 hand keypoints (wrist + 4 per finger).

    Keypoint order (MediaPipe / YOLO11n-pose-hands):
      0: wrist
      1-4: thumb (cmc, mcp, ip, tip)
      5-8: index (mcp, pip, dip, tip)
      9-12: middle (mcp, pip, dip, tip)
      13-16: ring (mcp, pip, dip, tip)
      17-20: pinky (mcp, pip, dip, tip)
    """

    Object: ClassVar[MetaObject] = KeypointObjectWithBbox

    labels: Optional[Union[tuple, list]] = field(default_factory=lambda: ['hand'])
    keypoints_shape = [21, 3]

    # Hand skeleton connections for rendering
    _point_bounds = [
        (0, 1, 2, 3, 4),        # thumb
        (0, 5, 6, 7, 8),        # index
        (0, 9, 10, 11, 12),     # middle
        (0, 13, 14, 15, 16),    # ring
        (0, 17, 18, 19, 20),    # pinky
        (5, 9, 13, 17),         # palm
    ]

    def draw(self, draw):
        from axelera.app.meta.base import draw_bounding_boxes
        draw_bounding_boxes(
            self,
            draw,
            self.task_render_config.show_labels,
            self.task_render_config.show_annotations,
        )

        if not self.task_render_config.show_annotations:
            return

        if len(self.keypoints) == 0:
            return

        _cyan = (0, 255, 255, 255)
        _yellow = (255, 255, 0, 255)

        lines = []
        for det_pts in self.keypoints:
            for pt_bound in self._point_bounds:
                line = [det_pts[kp][:2] for kp in pt_bound if det_pts[kp][2] > 0.3]
                if len(line) > 1:
                    lines.append(line)
            for x, y, v in det_pts:
                if v > 0.3:
                    draw.keypoint((x, y), _cyan, 4)
        if lines:
            draw.polylines(lines, False, _yellow, 2)

    @classmethod
    def decode(cls, data: Dict[str, Union[bytes, bytearray]]) -> Self:
        values_per_kpt_to_datatype = {
            2: np.dtype([('x', np.int32), ('y', np.int32)]),
            3: np.dtype([('x', np.int32), ('y', np.int32), ('visibility', np.float32)]),
        }

        boxes = decode_bbox(data)
        kpts = data.get("kpts", b"")
        if kpts_shape := data.get("kpts_shape", b""):
            [kpts_per_box, values_per_kpt] = np.frombuffer(kpts_shape, dtype=np.int32)
        else:
            values_per_kpt = 3
            kpts_per_box = 21

        kpts = np.frombuffer(kpts, dtype=values_per_kpt_to_datatype[values_per_kpt])
        kpts = np.vstack([kpts['x'], kpts['y']])
        if values_per_kpt == 3:
            vis = np.frombuffer(data.get("kpts", b""), dtype=values_per_kpt_to_datatype[3])
            kpts = np.vstack([kpts, vis['visibility']])
        kpts = kpts.T.astype(float).reshape(-1, kpts_per_box, values_per_kpt)

        scores = data.get('scores', b'')
        scores = np.frombuffer(scores, dtype=np.float32)
        return cls(keypoints=kpts, boxes=boxes, scores=scores)


class DecodeHandPose(DecodeYoloPose):
    """YOLO pose decoder for 21 hand keypoints."""

    # Re-declare parent annotations so AxOperator.__init_subclass__ picks them up
    box_format: str
    normalized_coord: bool
    conf_threshold: float = 0.25
    max_nms_boxes: int = 30000
    nms_iou_threshold: float = 0.65
    nms_top_k: int = 300

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        compiled_model_dir,
        task_graph,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, compiled_model_dir, task_graph
        )
        self.meta_type_name = "HandKeypointsMeta"
        self._kpts_shape = HandKeypointsMeta.keypoints_shape

    def exec_torch(self, image, predict, meta):
        from axelera.app.meta import BBoxState
        from axelera.app.torch_utils import torch

        if type(predict) == torch.Tensor:
            predict = predict.cpu().detach().numpy()

        if len(predict) == 1 and predict.shape[0] > 1:
            raise ValueError(
                f"Batch size >1 not supported, output tensor={predict[0].shape}"
            )

        bboxes = predict[0]
        if bboxes.shape[0] < bboxes.shape[1]:
            bboxes = bboxes.transpose()
        # 68 channels: 4 (box) + 1 (conf) + 63 (21 kpts × 3)
        kpts = bboxes[:, 5:]
        box_confidence = bboxes[:, 4]
        box_coordinates = bboxes[:, :4]

        indices = box_confidence > self.conf_threshold
        box_confidence = box_confidence[indices]
        box_coordinates = box_coordinates[indices]
        kpts = kpts[indices]

        if self._where:
            master_meta = meta[self._where]
            base_box = master_meta.boxes[
                master_meta.get_next_secondary_frame_index(self.task_name)
            ]
            src_img_width = base_box[2] - base_box[0]
            src_img_height = base_box[3] - base_box[1]
        else:
            src_img_width = image.size[0]
            src_img_height = image.size[1]

        state = BBoxState(
            self.model_width,
            self.model_height,
            src_img_width,
            src_img_height,
            self.box_format,
            self.normalized_coord,
            self.scaled,
            self.max_nms_boxes,
            self.nms_iou_threshold,
            nms_class_agnostic=self._nms_class_agnostic,
            output_top_k=self.nms_top_k,
        )

        if self._where:
            box_coordinates[:, [0, 2]] += base_box[0]
            box_coordinates[:, [1, 3]] += base_box[1]

        boxes, scores, kpts = state.organize_bboxes_and_kpts(
            box_coordinates, box_confidence, kpts
        )

        model_meta = HandKeypointsMeta(
            keypoints=kpts,
            boxes=boxes,
            scores=scores,
        )
        meta.add_instance(self.task_name, model_meta, self._where)
        return image, predict, meta
