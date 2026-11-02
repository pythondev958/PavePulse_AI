import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from pavepulse_core import (
    CATEGORY_ORDER, Detection, ReviewSegmenter,
    normalize_class, safe_bbox, summary_of_timeline,
)
from pavepulse_visuals import render_dashboard


class CoreTests(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(normalize_class('D00'), 'longitudinal_crack')
        self.assertEqual(normalize_class('D10'), 'transverse_crack')
        self.assertEqual(normalize_class('D20'), 'alligator_crack')
        self.assertEqual(normalize_class('D40'), 'pothole')
        self.assertEqual(normalize_class('pothole'), 'pothole')

    def test_invalid_generic_person_not_treated_as_road_damage(self):
        self.assertIsNone(normalize_class('person'))
        self.assertIsNone(normalize_class('car'))

    def test_bbox_clipped(self):
        self.assertEqual(safe_bbox((-10,10,200,55),100,80),(0,10,100,55))

    def test_bad_bbox_rejected(self):
        self.assertIsNone(safe_bbox((10,20,5,30),100,80))
        self.assertIsNone(safe_bbox((1,2,float('nan'),3),100,80))

    def test_review_segment_merges_short_gap(self):
        s=ReviewSegmenter(max_gap_frames=2,min_observed_frames=2)
        for i in range(7):
            detections=[Detection('pothole',0.70,(1,2,20,30))] if i in [0,1,4] else []
            s.update(i,detections)
        found=s.finalize()
        self.assertEqual(len(found),1)
        self.assertEqual(found[0].observed_frames,3)
        self.assertEqual((found[0].start_frame,found[0].end_frame),(0,4))

    def test_review_segment_split(self):
        s=ReviewSegmenter(max_gap_frames=1,min_observed_frames=2)
        for i in range(8):
            s.update(i,[Detection('pothole',0.7,(1,1,20,20))] if i in [0,1,5,6] else [])
        self.assertEqual(len(s.finalize()),2)

    def test_single_frame_not_a_review_interval(self):
        s=ReviewSegmenter(max_gap_frames=1,min_observed_frames=2)
        s.update(0,[Detection('pothole',0.7,(1,2,20,25))])
        s.update(1,[])
        self.assertEqual(len(s.finalize()),0)

    def test_summary_empty(self):
        self.assertEqual(summary_of_timeline([],[])['mean_simultaneous_detections'],0)

    def test_supported_model_label_sets(self):
        from pavepulse import supported_model_classes
        self.assertEqual(supported_model_classes({0: "Pothole"}), ("pothole",))
        self.assertEqual(set(supported_model_classes({0:"D00",1:"D10",2:"D20",3:"D40"})), set(CATEGORY_ORDER))
        with self.assertRaises(RuntimeError):
            supported_model_classes({0:"person",1:"car"})

    def test_dashboard_dimensions(self):
        frame=np.full((360,640,3),100,np.uint8)
        image=render_dashboard(frame,[Detection('pothole',0.91,(80,100,200,250))],
            8,0.3,25.0,100,{},[1,0,2,1],[(.1,'pothole',0.91)],0,'RDD2022','test.mp4')
        self.assertEqual(image.shape,(900,1600,3))
        self.assertGreater(float(image.std()),12)
        solo=render_dashboard(frame,[Detection('pothole',0.91,(80,100,200,250))],
            8,0.3,25.0,100,{},[1,0,2,1],[(.1,'pothole',0.91)],0,'Pothole specialist','test.mp4',('pothole',))
        self.assertEqual(solo.shape,(900,1600,3))


class VideoPipelineTest(unittest.TestCase):
    def test_smoke_generated_video_reports(self):
        from pavepulse import process
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'mock_source.mp4'
            w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),8,(320,180))
            self.assertTrue(w.isOpened())
            for i in range(5):
                frm=np.full((180,320,3),(34+i*8,46+i*3,58+i*3),np.uint8)
                cv2.line(frm,(0,90),(320,150),(110,110,115),3)
                w.write(frm)
            w.release()
            count=[0]
            def fake(_frame):
                count[0]+=1
                return [Detection('pothole',0.76,(60,55,140,110))] if count[0] <=3 else []
            results=process(path,Path(tmp)/'out',fake,keep_mp4v=True,model_tag='MOCK TEST ONLY')
            out=Path(tmp)/'out'
            self.assertEqual(results['frames_processed'],5)
            self.assertEqual(results['review_segments'],1)
            self.assertTrue((out/'pavepulse_annotated.mp4').is_file())
            self.assertTrue((out/'pavepulse_preview.jpg').is_file())
            self.assertTrue((out/'pavepulse_activity.png').is_file())
            self.assertEqual(len((out/'pavepulse_detections.csv').read_text(encoding='utf-8-sig').splitlines()),4)
            self.assertEqual(json.loads((out/'pavepulse_summary.json').read_text())['frame_level_detection_observations_total'],3)
            cap=cv2.VideoCapture(str(out/'pavepulse_annotated.mp4'))
            self.assertTrue(cap.isOpened())
            ret,frame=cap.read()
            cap.release()
            self.assertTrue(ret)
            self.assertEqual(frame.shape[:2],(900,1600))


if __name__=='__main__':
    unittest.main()
