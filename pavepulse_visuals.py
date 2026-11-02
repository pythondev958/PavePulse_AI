"""A widescreen field-inspection UI deliberately distinct from CrowdVision.

Visual design: pale diagnostic labels, warm pavement-gold highlights, a fixed
inspection feed at left, a narrow right-hand classification panel, and a
bottom activity strip. No fictional road-health score or severity estimation.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from pavepulse_core import (
    CATEGORY_ORDER,
    CLASS_COLORS,
    CLASS_SHORT,
    Detection,
)

CANVAS_W, CANVAS_H = 1600, 900
BG = (19, 23, 29)
PANEL = (31, 37, 43)
PANEL_2 = (41, 47, 54)
INK = (241, 243, 247)
MUTED = (158, 170, 180)
SILVER = (198, 207, 218)
ACCENT = (56, 200, 246)  # amber-ish BGR
TEAL = (177, 206, 91)


def put(img, text, x, y, scale=0.55, color=INK, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, str(text), (int(x), int(y)), font, float(scale), color, int(thickness), cv2.LINE_AA)


def rect(img, x1, y1, x2, y2, color, thickness=-1, radius=0):
    # Simple rectangular panels are intentional; corner radius not needed.
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, int(thickness), cv2.LINE_AA)


def mix(base: np.ndarray, overlay: np.ndarray, opacity: float):
    return cv2.addWeighted(overlay, opacity, base, 1 - opacity, 0)


def fit_frame(frame: np.ndarray, width: int, height: int):
    ih, iw = frame.shape[:2]
    ratio = min(width / iw, height / ih)
    w = max(1, round(iw * ratio))
    h = max(1, round(ih * ratio))
    im = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)
    pad = np.full((height, width, 3), (21, 25, 30), np.uint8)
    ox = (width - w) // 2
    oy = (height - h) // 2
    pad[oy:oy+h, ox:ox+w] = im
    return pad, (ox, oy, ratio)


def _corner_box(img, bbox, color, text):
    x1, y1, x2, y2 = bbox
    x1, y1 = int(x1), int(y1)
    x2, y2 = int(x2), int(y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    img[:] = mix(img, overlay, 0.68)
    extent = max(9, min(28, min(x2-x1, y2-y1)//3))
    for ax, ay, dx, dy in [(x1,y1,1,1), (x2,y1,-1,1), (x1,y2,1,-1), (x2,y2,-1,-1)]:
        cv2.line(img, (ax, ay), (ax + dx*extent, ay), color, 3, cv2.LINE_AA)
        cv2.line(img, (ax, ay), (ax, ay + dy*extent), color, 3, cv2.LINE_AA)
    scale=0.42
    text_width, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0]
    tx = max(1, min(img.shape[1]-text_width-12, x1))
    ty = max(th+10, y1-8)
    rect(img, tx, ty-th-9, tx+text_width+12, ty+6, (20, 24, 30))
    rect(img, tx, ty-th-9, tx+3, ty+6, color)
    put(img, text, tx+7, ty, scale, INK, 1)


def draw_road_frame(video: np.ndarray, dets: List[Detection], frame_index: int):
    vh,vw = video.shape[:2]
    annotated = video.copy()
    # Diagnostic grid and sweep are aesthetic; they are not road sensors.
    overlay = annotated.copy()
    for y in range(0, vh, 110):
        cv2.line(overlay, (0, y), (vw, y), (110, 120, 129), 1)
    for x in range(0, vw, 130):
        cv2.line(overlay, (x, 0), (x, vh), (110, 120, 129), 1)
    annotated = mix(annotated, overlay, 0.075)
    for d in dets:
        _corner_box(annotated, d.bbox, CLASS_COLORS[d.cls], f"{CLASS_SHORT[d.cls]}  {d.confidence:.0%}")
    # Upper frame labels are away from most road-surface foreground detections.
    overlay = annotated.copy()
    rect(overlay, 12, 14, 225, 54, (24, 28, 34))
    annotated = mix(annotated, overlay, 0.82)
    rect(annotated, 19, 22, 24, 45, ACCENT)
    put(annotated, "SURFACE INSPECTION", 35, 39, 0.54, INK, 1)
    overlay = annotated.copy()
    rect(overlay, vw-182, 16, vw-10, 48, (24,28,34))
    annotated = mix(annotated, overlay, 0.83)
    put(annotated, "AI PREDICTIONS", vw-172, 37, 0.47, (205,215,224))
    return annotated


def render_dashboard(frame: np.ndarray, dets: List[Detection], frame_id: int,
                     seconds: float, fps: float, total_frames: int, counts_total: dict,
                     history: Iterable[int], last_seen: List[Tuple[float,str,float]],
                     segment_count: int, model_tag: str, source_name: str,
                     supported_classes=CATEGORY_ORDER):
    canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, np.uint8)
    rect(canvas, 0, 0, CANVAS_W, 89, (25, 29, 35))
    rect(canvas, 0, 87, CANVAS_W, 90, (48, 55, 62))
    rect(canvas, 0, 0, 9, 91, ACCENT)
    put(canvas, "PAVEPULSE", 29, 43, 1.08, INK, 2)
    put(canvas, "AI", 248, 43, 0.75, ACCENT, 2)
    put(canvas, "ROAD CONDITION / VISUAL INSPECTION", 29, 69, 0.47, MUTED)
    put(canvas, "RESEARCH PROTOTYPE", 1155, 36, 0.57, ACCENT, 2)
    put(canvas, "MODEL-ASSISTED  |  HUMAN REVIEW REQUIRED", 1155, 67, 0.43, SILVER)

    vx, vy, vw, vh = 27, 111, 1118, 634
    cv2.rectangle(canvas, (vx-2,vy-2), (vx+vw+2,vy+vh+2), (66,72,79), 1)
    viewport, (ox,oy,ratio) = fit_frame(frame, vw, vh)
    transformed = []
    for d in dets:
        b=(int(d.bbox[0]*ratio)+ox, int(d.bbox[1]*ratio)+oy,
           int(d.bbox[2]*ratio)+ox, int(d.bbox[3]*ratio)+oy)
        transformed.append(Detection(d.cls, d.confidence, b))
    viewport = draw_road_frame(viewport, transformed, frame_id)
    canvas[vy:vy+vh, vx:vx+vw] = viewport

    # Right rail: use compact class-list metrics rather than floating HUD cards.
    sx, sy, sw, sh = 1165, 111, 407, 634
    rect(canvas, sx, sy, sx+sw, sy+sh, PANEL)
    rect(canvas, sx, sy, sx+sw, sy+8, ACCENT)
    put(canvas, "INSPECTION READOUT", sx+21, sy+39, 0.64, INK, 2)
    put(canvas, "Frame-level model observations", sx+21, sy+66, 0.46, MUTED)
    rect(canvas, sx+21, sy+81, sx+sw-22, sy+82, (69,76,83))
    current = len(dets)
    put(canvas, "VISIBLE PROPOSALS", sx+21, sy+114, 0.47, MUTED)
    put(canvas, f"{current:02d}", sx+21, sy+178, 1.92, INK, 3)
    put(canvas, "POSSIBLE DEFECTS", sx+136, sy+151, 0.49, ACCENT)
    put(canvas, "IN THIS FRAME", sx+136, sy+173, 0.47, MUTED)
    rect(canvas, sx+21, sy+194, sx+sw-21, sy+195, (69,76,83))
    pothole_only = tuple(supported_classes) == ("pothole",)
    put(canvas, "POTHOLE MODEL" if pothole_only else "DETECTION MIX", sx+21, sy+224, 0.52, SILVER, 1)
    current_counts={k:0 for k in CATEGORY_ORDER}
    for d in dets: current_counts[d.cls]+=1
    max_count=max(1, max(current_counts.values()))
    for i, cls in enumerate(("pothole",) if pothole_only else CATEGORY_ORDER):
        y=sy+263+i*62
        col=CLASS_COLORS[cls]
        cv2.circle(canvas, (sx+30,y-7), 6, col, -1, cv2.LINE_AA)
        put(canvas, CLASS_SHORT[cls], sx+47, y-2, 0.49, INK)
        put(canvas, f"{current_counts[cls]:02d}", sx+sw-64,y, 0.70, INK,2)
        rect(canvas,sx+47,y+11,sx+sw-25,y+16,(65,69,76))
        if current_counts[cls]>0:
            rect(canvas,sx+47,y+11,sx+47+int((sw-72)*current_counts[cls]/max_count),y+16,col)
    if pothole_only:
        put(canvas, "Pothole-only detector active", sx+47, sy+334, 0.48, SILVER)
        put(canvas, "Crack types NOT evaluated", sx+47, sy+370, 0.44, MUTED)
        put(canvas, "No GPS or physical depth estimate", sx+47, sy+409, 0.39, MUTED)
    rect(canvas,sx+21,sy+501,sx+sw-22,sy+502,(69,76,83))
    put(canvas, "RECENT OBSERVATIONS", sx+21, sy+528, 0.47, SILVER)
    recent=last_seen[-2:]
    if not recent:
        put(canvas,"No model detections yet",sx+21,sy+556,0.45,MUTED)
    for i, (tm,cls,conf) in enumerate(reversed(recent)):
        y=sy+558+i*30
        put(canvas,f"{tm:05.1f}s",sx+21,y,0.46,MUTED)
        put(canvas,f"{CLASS_SHORT[cls]}  {conf:.0%}",sx+100,y,0.46,CLASS_COLORS[cls])
    put(canvas,"Review segments (not unique defects)",sx+21,sy+616,0.38,MUTED)
    put(canvas,f"{segment_count:02d} finalized",sx+sw-129,sy+616,0.41,ACCENT)

    # Bottom strip is a slim seismograph-style observation trace, not a severity score.
    rect(canvas, 27, 764, 1572, 873, (27,32,39))
    put(canvas, "ACTIVITY  /  DETECTIONS PER FRAME", 42, 790, 0.45, SILVER)
    put(canvas, f"{seconds:05.1f}s", 1428, 790, 0.63, INK, 1)
    put(canvas, f"{fps:.1f} fps source", 1299, 816, 0.43, MUTED)
    history=list(history)
    gx1,gy1,gx2,gy2=326,796,1257,849
    cv2.line(canvas, (gx1,gy2), (gx2,gy2), (82,89,96), 1)
    cv2.line(canvas, (gx1,gy1), (gx2,gy1), (51,59,67), 1)
    if history:
        shown=history[-180:]
        m=max(1, max(shown))
        points=[]
        for i,v in enumerate(shown):
            xx=gx1+int(i*max(1,gx2-gx1)/max(1,len(shown)-1))
            yy=gy2-int(v/m*(gy2-gy1))
            points.append((xx,yy))
        if len(points)>1:
            cv2.polylines(canvas,[np.array(points,np.int32)],False,ACCENT,2,cv2.LINE_AA)
        for xx,yy in points[::max(1,len(points)//15)]:
            cv2.circle(canvas,(xx,yy),2,ACCENT,-1,cv2.LINE_AA)
    if total_frames>0:
        done=min(1, (frame_id+1)/total_frames)
        rect(canvas, 27, 882, 1572, 887, (56,62,70))
        rect(canvas, 27, 882, 27+int(done*(1572-27)), 887, ACCENT)
    put(canvas,"IMAGE-SPACE RESULTS  |  NOT GPS / SEVERITY / UNIQUE DEFECT COUNTS",42,861,0.36,MUTED)
    return canvas
