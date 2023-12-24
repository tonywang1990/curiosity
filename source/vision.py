from dataclasses import dataclass
import os
import cv2
from PIL import Image
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference
import logging
import collections
from source.primitive import Target, Object, Context

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    default_model_dir: str
    default_model: str


@dataclass
class ODConfig(ModelConfig):
    default_model_dir: str = 'resource/' # /home/tonywy/SunFounder_PiCar-V/
    default_model: str = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels: str = 'coco_labels.txt'
    model: str = os.path.join(default_model_dir, default_model)
    labels: str = os.path.join(default_model_dir, default_labels)
    top_k: int = 10
    threshold: float = 0.1
    camera_idx: int = 0


class ObjectDetection:
    def __init__(self, config: ODConfig):
        self.config = config
        self._setup(config)

    def _setup(self, config: ODConfig):
        logger.debug('Loading {} with {} labels.'.format(
            config.model, config.labels))
        self.interpreter = make_interpreter(config.model)
        self.interpreter.allocate_tensors()
        self.labels = read_label_file(config.labels)
        self.inference_size = input_size(self.interpreter)

    def detect(self, frame: np.ndarray) -> List[Object]:
        interpreter = self.interpreter
        config = self.config
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        logger.debug(
            f'frame size {frame.shape}, inference size {self.inference_size}')
        frame = cv2.resize(frame, self.inference_size)
        run_inference(interpreter, frame.tobytes())
        objs = get_objects(interpreter, config.threshold)[:config.top_k]
        return objs
        # cv2_im = append_objs_to_img(cv2_im, inference_size, objs, self.labels)

class Cortex:
    def __init__(self, configs: List[ModelConfig]):
        self.object_detection = None
        for config in configs:
            if isinstance(config, ODConfig):
                self.object_detection = ObjectDetection(config)

    def process(self, frame: np.ndarray, context: Context, args: Dict[str, Any]) -> Context:
        if self.object_detection:
            context = self.process_object_detection(frame, context, args)
        return context

    def process_object_detection(self, frame: np.ndarray, context: Context, args: Dict[str, Any]) -> Context:
        assert self.object_detection
        assert 'tgt_ids' in args
        objs = self.object_detection.detect(frame)

        height, width, _ = frame.shape
        scale_x, scale_y = width / self.object_detection.inference_size[0], height / self.object_detection.inference_size[1]
        objs = self.scale_objs(objs, (scale_x, scale_y))

        tgt = find_target(objs, args)
        context.target = tgt
        return context

    def scale_objs(self, objs: List[Object], scale: Tuple[int]) -> List[Object]:
        scaled = []
        for obj in objs:
            scaled.append(Object(id=obj.id, score = obj.score, bbox=obj.bbox.scale(scale[0], scale[1])))
        return scaled
    
    def get_target_list(self):
        return self.object_detection.labels


def find_target(objs: List[Object], args: Dict[str, Any]) -> Optional[Target]:
    def _objects_to_targets(objs: List[Object]) -> List[Target]:
        targets = []
        for obj in objs:
            bbox = obj.bbox
            x0, y0 = int(bbox.xmin), int(bbox.ymin)
            x1, y1 = int(bbox.xmax), int(bbox.ymax)
            x_center, y_center = (x0+x1)/2, (y0+y1)/2
            width, height = abs(x1-x0), abs(y1-y0)
            targets.append(
                Target(obj.id, obj.score, obj.bbox, (x_center, y_center), (width, height)))
        return targets

    def _filter_by_id(objects: List[Object], ids: List[int]) -> List[Object]:
        return [obj for obj in objects if obj.id in ids]

    def _filter_by_max(tgts: List[Target], key: str) -> Optional[Target]:
        def _get_key(tgt: Target) -> float:
            if key == 'score':
                return tgt.score
            elif key == 'size':
                return tgt.size
            else:
                raise NotImplementedError
        if len(tgts) == 0:
            return None
        return max(tgts, key=_get_key)

    if 'tgt_ids' in args:
        _objs = _filter_by_id(objs, args['tgt_ids'])
    else:
        _objs = objs
    tgts = _objects_to_targets(_objs)
    if 'tgt_key' in args:
        tgt = _filter_by_max(tgts, args['tgt_key'])
    else:
        logger.warning(
            f'Multiple target found without tgt_key, returning the first one, args: {args}')
        tgt = tgts[0]
    if tgt is None:
        logger.warning(
            f'Found 0 targets, there are {len(objs)} objects, args: {args}')
    return tgt