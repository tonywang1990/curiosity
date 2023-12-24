import collections
from pycoral.adapters.detect import Object, BBox
from typing import Optional
import json

Target = collections.namedtuple('Target', ['id', 'score', 'bbox', 'center', 'size'])

FrameSize = collections.namedtuple('FrameSize', ['h', 'w'])

class Context:
    target: Optional[Target] = None
    camera_pan: Optional[int] = None
    camera_tilt: Optional[int] = None
    frame_size: Optional[int] = None
    turn: Optional[int] = None
    speed: Optional[int] = None

    def __str__(self) -> str:
        return json.dumps(self.__dict__, indent = 2)
