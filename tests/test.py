from source.car import Camera, Car, CameraServo, FrontWheels, BackWheels
from source.vision import ODConfig, ObjectDetection, find_target
from source.primitive import FrameSize, Object, BBox, Target, Context
import unittest
import time
import numpy as np
import cv2

# The code you want to test goes in a separate module or class
# For example, let's create a simple function to test


# Now, create a test class that inherits from unittest.TestCase

class TestCamera(unittest.TestCase):
    def setUp(self):
        self.frame_size = FrameSize(h=480, w=640)
        self.camera = Camera(frame_size=self.frame_size)
        self.camera_servo = self.camera.servo
        self.front_wheels = FrontWheels()
        self.back_wheels = BackWheels()
        self.dummy_bbox = BBox(1,2,3,4)

    # Each test is a method that starts with the word "test"
    def test_front_wheels(self):
        self.front_wheels.reset()
        self.front_wheels.turn = 85.5
        self.assertEqual(self.front_wheels.turn, 85.5)
        time.sleep(0.3)
        self.front_wheels.turn = 35
        self.assertEqual(self.front_wheels.turn, max(35, self.front_wheels._min_angle))
    
    def test_back_wheels(self):
        self.back_wheels.reset()
        self.back_wheels.speed = 50
        self.assertEqual(self.back_wheels.speed, 50)
        time.sleep(2)
        self.back_wheels.speed = -50
        self.assertEqual(self.back_wheels.speed, -50)
        time.sleep(2)
        self.back_wheels.speed = 100
        self.assertEqual(self.back_wheels.speed, self.back_wheels._max_speed)

    def test_back_wheel_follow_target(self):
        context = Context()
        context.frame_size = (100,100)
        context.target = Target(id=1, score=0, bbox=self.dummy_bbox, center=(0,0), size=(20,20))
        speed = self.back_wheels.track_target(context, goal_ratio=0.5)
        self.assertTrue(speed >= 0)
        #context.target = Target(id=1, score=0, bbox=self.dummy_bbox, center=(0,0), size=(80,80))
        #speed = self.back_wheels.track_target(context, goal_ratio=0.5)
        #self.assertTrue(speed <= 0)

    def test_camera_servo(self):
        self.assertEqual(self.camera_servo.pan, 90)
        self.camera_servo.pan = 80
        self.camera_servo.tilt = 100
        self.assertEqual(self.camera_servo.pan, 80)
        time.sleep(0.5)
        self.camera_servo.reset()
        self.assertEqual(self.camera_servo.pan, 90)
        self.assertEqual(self.camera_servo.tilt, 90)
    
    def test_camera_capture(self):
        self.assertTrue(self.camera.isOpened())
        ret, img = self.camera.read()
        self.assertTrue(ret)
        self.assertEqual(img.shape, self.frame_size + (3,)) # append channel
        self.assertNotEqual(np.mean(img), 255)
        self.assertNotEqual(np.mean(img), 0)
        self.assertNotEqual(np.std(img), 0)
    
    def test_camera_track_target(self):
        context = Context()
        context = self.camera.follow_target(context)
        self.assertEqual(context.camera_pan, 90)
        dummy_bbox = BBox(1,2,3,4)
        target=Target(id=0, score=1.0, bbox=dummy_bbox, center=(100, 500), size = 3000)
        direction = self.camera.track_target(target)
        self.assertTrue(direction[0] >= 0 and direction[1] <= 0)
        #target=Target(id=0, score=1.0, bbox=dummy_bbox, center=(400, 200), size = 3000)
        #direction = self.camera.track_target(target)
        #self.assertTrue(direction[0] <= 0 and direction[1] >= 0)

    def test_vision_model(self):
        test_img_path = 'resource/kite_and_cold.jpg'
        img = cv2.imread(test_img_path)
        model = ObjectDetection(ODConfig())
        objs = model.detect(img)
        self.assertTrue(len(objs) != 0)
        ids = []
        for obj in objs:
            ids.append(obj.id)
        self.assertTrue(37 in ids)
    
    def test_find_target(self):
        objs = [Object(id=0, score=0.1, bbox=BBox(0, 0, 5, 5)), 
                Object(id=0, score=0.9, bbox=BBox(0, 0, 3, 3)),
                Object(id=1, score=0.2, bbox=BBox(0, 0, 6, 7)),
                Object(id=1, score=0.8, bbox=BBox(0, 0, 4, 5)),
                ]
        tgt = find_target(objs, args={'tgt_ids': [0]})
        self.assertTrue(tgt is not None)

        tgt = find_target(objs, args={'tgt_ids': [0, 1], 'tgt_key': 'size'})
        self.assertTrue(tgt.score == 0.2)

        tgt = find_target(objs, args={'tgt_ids': [0, 1], 'tgt_key': 'score'})
        self.assertTrue(tgt.score == 0.9)

        tgt = find_target(objs, args={'tgt_ids': [0], 'tgt_key': 'score'})
        self.assertTrue(tgt.score == 0.9)


if __name__ == '__main__':
    unittest.main()
