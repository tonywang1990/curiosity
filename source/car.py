import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any, List, Optional
from picar.SunFounder_PCA9685 import Servo
import picar
import logging
from primitive import Target, FrameSize, Context
from vision import Cortex, ODConfig
import time
from collections import deque
from enum import Enum
from pid import PID
import matplotlib.pyplot as plt
import traceback



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  

def no_action_window(value: float, lower: float, upper: float, default_value: float) -> float:
    if value >= lower and value <=upper:
        value = default_value
    return value

def plot_curve(data: List[float], label: str) -> None:
    indices = range(1, len(data) + 1)

    # Plot the curve
    plt.plot(indices, data, marker='o', linestyle='-', color='b')

    # Set labels and title
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title(label)

    # Show the plot
    plt.show()

class Controller:
    def __init__(self, kp: float, ki: float, kd: float, setpoint: float = 0.0):
        self._impl = PID(kp, ki, kd, setpoint)
        self.log = {'error': [], 'control': []}
    
    def __call__(self, error: float) -> float:
        self.log['error'].append(error)
        control = self._impl(error)
        self.log['control'].append(control)
        return control
    
    def plot(self, label: str, key: str = 'error') -> None:
        plot_curve(self.log[key], f"{label}.{key}")

    

class CameraServo:
    def __init__(self) -> None:
        # Control.
        self._pan_servo = Servo.Servo(1, bus_number=1)
        self._tilt_servo = Servo.Servo(2, bus_number=1)
        self.setup_servo()

    
    ## Servo methods.
    def setup_servo(self):
        self._pan_servo.setup()
        self._tilt_servo.setup()
        # Log initial angels.
        self._pan_angle = 90
        self._tilt_angle = 90

    def regularize(self, angle: int) -> int:
        if angle > 180:
            angle = 180
        if angle < 0:
            angle = 0
        return angle

    @property
    def pan(self):
        """
        facing camera:
            left most: 0
            right most: 180
        """
        return self._pan_angle

    @pan.setter
    def pan(self, angle: int):
        angle = self.regularize(angle)
        self._pan_servo.write(angle)
        self._pan_angle = angle

    @property
    def tilt(self):
        """
        facing camera:
            down: 0
            up: 180
        """
        return self._tilt_angle

    @tilt.setter
    def tilt(self, angle: int):
        angle = self.regularize(angle)
        self._tilt_servo.write(angle)
        self._tilt_angle = angle
    
    def reset(self) -> None:
        self.pan = 90
        self.tilt = 90  

class Camera:
    def __init__(self, frame_size: FrameSize) -> None:
        """
        frame_size: H x W
        """
        self._camera = cv2.VideoCapture(0)
        logger.info(f"camera frame rate: {self._camera.get(cv2.CAP_PROP_FPS)}")
        self._frame_size = frame_size
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size.w)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size.h)
        self.servo = CameraServo()
        # Determined by Ziegler-Nichols method:
        # Kpu = 0.038, Tu = 23 /30 / 30 = 0.0255
        # Kp = 0.6 * Kpu = 0.0228
        # Ki = 0.5 * Tu = 0.0122
        # Kd = 0.125 * Tu = 0.0031
        kp, ki, kd = 0.038, 0.0002, 0.002
        self._pan_control = Controller(kp, ki, kd)
        self._tilt_control = Controller(kp, ki, kd)

    def track_target(self, target: Target) -> Tuple[int, int]:
        assert target is not None, target
        frame_center = self._frame_size.w / 2, self._frame_size.h / 2
        error = target.center[0] - frame_center[0], target.center[1] - frame_center[0] 
        update = self._pan_control(error[0]), self._tilt_control(error[1])
        logger.info(f"Camera: frame center {frame_center}, target center {target.center}, error {error}, update {update}")
        return update

    def follow_target(self, context: Context) -> Context:
        """
        Move camera servo to track target. 
        Servo is moved such that frame center is close to target center.
        """
        target = context.target
        if target is None:
            logger.warning(f"{self.__class__}: target is None, maintain currnet angle..")
        else:
            update = self.track_target(target)
            logger.info(f"pan: {self.servo.pan}, pan_change: {update[0]}, tilt: {self.servo.tilt}, pan_change: {update[1]}")
            self.servo.pan += update[0]
            self.servo.tilt += update[1]
        context.camera_pan = self.servo.pan
        context.camera_tilt = self.servo.tilt
        context.frame_size = self._frame_size
        return context

    def isOpened(self) -> bool:
        return self._camera.isOpened()

    def read(self) -> np.ndarray:
        return self._camera.read()

    def capture(self, format: str = 'BGR') -> np.ndarray:
        _, frame = self.read()
        if format == 'RGB':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif format == 'BGR':
            pass
        else:
            raise NotImplementedError
        return frame

    #def display_capture(self):
    #    frame = self.capture('RGB')
    #    display(Image.fromarray(frame))

    # Override to make sure camera instance is released after finished.
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.reset()

    def reset(self):
        self.servo.reset()
        self._camera.release()

class FrontWheels:
    def __init__(self, turning_offset: int = 0, max_turn: int = 35):
        self._servo = Servo.Servo(channel=0, bus_number=1, offset=turning_offset)
        self._max_turn = max_turn
        self._angle = 90
        self._min_angle = 90 - self._max_turn
        self._max_angle = 90 + self._max_turn
    
    def reset(self):
        self.turn = 90
    
    @property
    def turn(self):
        """
            facing the front_wheels:
                left most: 180
                right most: 0
        """
        return self._angle

    @turn.setter
    def turn(self, angle: int):
        angle = min(max(self._min_angle, angle), self._max_angle)
        self._servo.write(angle)
        self._angle = angle
    
    def follow_target(self, context: Context) -> Context:
        try:
            if context.camera_pan and context.speed:
                #    turn = pan_turn_map * camera_pan_angle * move_direction
                turn = 180 - context.camera_pan 
                if context.speed < 0:
                    turn = 180 - turn
                self.turn = turn
                logger.info(f"FrontWheels: direction {context.speed}, turn {self.turn}")
            else:
                logger.warning(f'FrontWheels: "camera_pan" not found in info, do nothing.. \n {context}')
            context.turn = self.turn
        except Exception as e:
            logger.warning(e)
        return context

class BackWheels:
    def __init__(self):
        self._back_wheels = picar.back_wheels.Back_Wheels()
        self._min_speed = -100
        self._max_speed = 100
        self._speed = 0
        #kp, ki, kd = 0.038, 0.0002, 0.002
        self._control = Controller(0.006, 0.001, 0)
    
    def reset(self):
        self._back_wheels.stop()
    
    @property
    def speed(self):
        """
            facing the front_wheels:
                left most: 180
                right most: 0
        """
        return self._speed

    @speed.setter
    def speed(self, value: int):
        value = min(max(self._min_speed, value), self._max_speed)
        self._speed = value
        self._back_wheels.speed = abs(self._speed)
        if self._speed > 0:
            self._back_wheels.forward()
        if self._speed < 0:
            self._back_wheels.backward()
    
    def follow_target(self, context:Context, goal_ratio: float ) -> Context:
        speed = self.track_target(context, goal_ratio)
        self.speed = speed
        context.speed = speed
        return context 

    def track_target(self, context:Context, goal_ratio: float ) -> int:
        frame_size = context.frame_size
        target = context.target
        if frame_size and target:
            size = frame_size[0] * frame_size[1] 
            actual_size = target.size[0] * target.size[1]
            goal_size = size * goal_ratio
            error = actual_size - goal_size 
            speed = int(self._control(error))
            #speed = -1 * error // 100
            #speed = no_action_window(-1 * delta // 100, -20, 20, 0)
            logger.info(f"Backwheels: actual_size {actual_size}, goal_size {goal_size}, error {error}, speed {self.speed}")
        else:
            logger.warning(f'Backwheels: no frame_size and target found, stoping.. \n {context}')
            speed = 0
        return speed

                
    def __exit__(self):
        self.reset()

class Car:
    def __init__(self):
        self.front_wheels = FrontWheels()
        self.back_wheels = BackWheels()
        self.setup()

        self.camera = Camera(frame_size=FrameSize(h=480, w=640))
        self.cortex = Cortex([ODConfig()])
        self.controllers = {'camera_pan': self.camera._pan_control, 'camera_tilt': self.camera._tilt_control, 'back_wheels': self.back_wheels._control}

    def setup(self):
        self.reset()

    def reset(self):
        self.front_wheels.reset()
        self.back_wheels.reset()
    
    def get_target_list(self):
        return self.cortex.get_target_list()

    def loop(self, args: Dict[str, Any], max_iter:int = 100, debug: bool = False):
        info, img = None, None
        camera = self.camera
        front_wheels = self.front_wheels
        back_wheels = self.back_wheels
        context = Context()
        try:
            for _ in range(max_iter):
                if not camera.isOpened():
                    logger.warning("camera.isOpened() == False, exiting loop..")
                    break
                ret, frame = camera.read()
                if not ret:
                    logger.warn('camera returned none, exiting loop..')
                    break
                context = self.cortex.process(frame, context, args)
                if debug:
                    img = plot_targets(frame, context.target, self.cortex.object_detection.labels)
                context = camera.follow_target(context)
                context = back_wheels.follow_target(context, goal_ratio = 0.3)
                context = front_wheels.follow_target(context)
                #time.sleep(0.001)
        except Exception as e:
            logger.warning(f'{e}')
            traceback.print_exc()

        finally:
            camera.reset()
            front_wheels.reset()
            back_wheels.reset()
        return context, img
    
    def test(self):
        #self.test_front_wheels()
        self.camera_servo.pan = 30
        self.camera_servo.tilt = 90
        time.sleep(0.5)
        self.camera_servo.reset()

    def test_front_wheels(self):
        picar.front_wheels.test()

    def test_back_wheels(self):
        picar.back_wheels.test()

def append_objs_to_img(cv2_im, objs, labels):
    #height, width, channels = cv2_im.shape
    #scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        bbox = obj.bbox
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im


def plot_targets(frame: np.ndarray, tgt: Target, labels) -> np.ndarray:
    if tgt is None:
        return None
    #height, width, channels = frame.shape
    #scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    # Choose dot properties
    dot_color = (0, 0, 255)  # Red color in BGR
    dot_radius = 5
    # Coordinates (x, y) for the center of the dot
    dot_center = (int(tgt.center[0]), int(tgt.center[1]))
    # Draw the dot on the image
    cv2.circle(frame, dot_center, dot_radius,
            dot_color, -1)  # -1 fills the circle
    append_objs_to_img(frame, [tgt], labels)
    return frame


