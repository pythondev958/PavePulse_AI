"""PavePulse AI — road-damage inspection video and review reports.

Examples:
 python pavepulse.py --input input/road.mp4 --device cpu --output outputs --show
 python pavepulse.py --input input/road.mp4 --max-seconds 5
 python pavepulse.py --input input/road.mp4 --weights models/best.pt --imgsz 960

The pretrained weights are a third-party Hugging Face model. Results are
experimental screen-space predictions, not GPS surveys, verified severity,
or deduplicated physical road defects.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import deque
from pathlib import Path
from typing import Callable, List

import cv2
import numpy as np

from pavepulse_core import (
    CATEGORY_ORDER,
    CLASS_TITLES,
    Detection,
    ReviewSegmenter,
    normalize_class,
    safe_bbox,
    summary_of_timeline,
)
from pavepulse_visuals import CANVAS_H, CANVAS_W, render_dashboard


def read_arguments(argv=None):
    p = argparse.ArgumentParser(
        description="PavePulse AI | YOLOv8 road defect detection with an inspection-themed annotated video"
    )
    p.add_argument("--input", required=True, help="Input road video path (.mp4 preferred)")
    p.add_argument("--output", default="outputs", help="Output directory")
    p.add_argument("--weights", default="models/pothole_specialist.pt", help="Pothole specialist weights (default) or models/best.pt for four-class RDD2022")
    p.add_argument("--device", default="cpu", help="Ultralytics device: cpu or 0 for first compatible GPU")
    p.add_argument("--imgsz", type=int, default=640, help="Model image size; try 960 for small cracks if CPU permits")
    p.add_argument("--conf", type=float, default=0.25, help="Detection-confidence threshold, default 0.25")
    p.add_argument("--analysis-width", type=int, default=1600, help="Resize original frames to at most this width before detection")
    p.add_argument("--max-seconds", type=float, default=0, help="Limit first N seconds of input; 0 for whole video")
    p.add_argument("--show", action="store_true", help="Show rendered frames on Windows (press Q or Esc to stop)")
    p.add_argument("--keep-mp4v", action="store_true", help="Skip automatic H.264 conversion; useful for troubleshooting")
    args=p.parse_args(argv)
    if args.conf <= 0 or args.conf > 1:
        p.error("--conf must be between 0 and 1")
    if args.imgsz < 320 or args.analysis_width < 320:
        p.error("--imgsz and --analysis-width must be at least 320")
    if args.max_seconds < 0:
        p.error("--max-seconds must be >= 0")
    return args


def supported_model_classes(model_names):
    """Only accept validated road-defect label sets; avoid generic COCO models."""
    names = model_names.values() if isinstance(model_names, dict) else model_names
    normalized = {normalize_class(name) for name in names}
    if normalized == {"pothole"}:
        return ("pothole",)
    if set(CATEGORY_ORDER).issubset(normalized):
        return CATEGORY_ORDER
    raise RuntimeError(
        f"Selected model has classes {model_names}. Expected pothole-only or all four RDD2022 classes. "
        "Generic COCO YOLO weights cannot detect road damage."
    )


def get_detector(weights: str):
    """Load either a pothole-specialist model or the original four-class RDD model."""
    from ultralytics import YOLO
    model_file = Path(weights)
    if not model_file.is_file():
        if model_file.as_posix().replace('\\','/') == 'models/pothole_specialist.pt':
            from huggingface_hub import hf_hub_download
            import shutil
            print("Pothole specialist model not found; downloading from Hugging Face...")
            model_file.parent.mkdir(parents=True, exist_ok=True)
            model_url = hf_hub_download(repo_id='Samdutse/pothole-yolov8', filename='best.pt')
            shutil.copy2(model_url, model_file)
        elif model_file.as_posix().replace('\\','/') == 'models/best.pt':
            from download_model import download_weights
            print("RDD2022 four-class model not found; downloading from Hugging Face...")
            model_file = download_weights('models')
        else:
            raise FileNotFoundError(f"Model weights not found: {model_file}")
    model = YOLO(str(model_file))
    supported_model_classes(model.names)
    return model


def infer_detections(model, frame, device="cpu", imgsz=640, conf=0.25) -> List[Detection]:
    result=model.predict(frame,device=device,imgsz=imgsz,conf=conf,max_det=100,verbose=False)[0]
    if result.boxes is None:
        return []
    h,w=frame.shape[:2]
    names=model.names
    output=[]
    for box in result.boxes:
        class_id=int(box.cls[0].item())
        original=names[class_id] if isinstance(names,(dict,list,tuple)) else str(class_id)
        cls=normalize_class(original)
        if not cls:
            continue
        bbox=safe_bbox(box.xyxy[0].tolist(),w,h)
        if bbox is None:
            continue
        output.append(Detection(cls,float(box.conf[0].item()),bbox))
    return output


def ensure_writer(path: Path, fps: float):
    codec=cv2.VideoWriter_fourcc(*"mp4v")
    writer=cv2.VideoWriter(str(path), codec, fps, (CANVAS_W, CANVAS_H))
    if not writer.isOpened():
        raise RuntimeError(
            f"Unable to open MP4 writer at {path}. Check OpenCV installation and folder access."
        )
    return writer


def convert_h264(working: Path, final: Path) -> str:
    """Create H.264/yuv420p MP4 suitable for major social sites, or keep fallback."""
    try:
        import imageio_ffmpeg
        import subprocess
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
        cmd=[str(ffmpeg),'-hide_banner','-loglevel','error','-y','-i',str(working),
             '-an','-c:v','libx264','-pix_fmt','yuv420p','-preset','fast','-crf','23',
             '-movflags','+faststart',str(final)]
        subprocess.run(cmd,check=True,timeout=600,capture_output=True,text=True)
        if not final.is_file() or final.stat().st_size < 500:
            raise RuntimeError("Output H.264 file missing/empty")
        working.unlink(missing_ok=True)
        return "H.264 / yuv420p"
    except Exception as exc:
        print(f"WARNING: H.264 conversion unavailable ({exc}); retaining MPEG-4 Part 2 fallback.")
        # Rename the valid working mp4; conversion failure is not video-analysis failure.
        if final.exists():
            final.unlink()
        working.replace(final)
        return "MPEG-4 Part 2 (mp4v); convert to H.264 for some social networks"


def write_csv(file: Path, rows: list, fields: list):
    with open(file,"w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def create_activity_chart(timeline: list, outfile: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax=plt.subplots(figsize=(12,4.2),dpi=130)
    fig.patch.set_facecolor('#151a21')
    ax.set_facecolor('#1f252d')
    if timeline:
        xs=[r['seconds'] for r in timeline]
        ys=[r['visible_detections'] for r in timeline]
        ax.fill_between(xs,ys,0,color='#f3bc38',alpha=.20)
        ax.plot(xs,ys,color='#f3bc38',lw=1.8,label='Frame-level detections')
        ax.set_xlim(0,max(0.1,xs[-1]))
        ax.set_ylim(0,max(2,max(ys)+1))
    ax.set_xlabel('Video time (seconds)', color='#b8c3cc')
    ax.set_ylabel('Model detections in frame', color='#b8c3cc')
    ax.set_title('PAVEPULSE AI  |  ROAD DEFECT OBSERVATION ACTIVITY', color='white', loc='left', fontsize=13,pad=15,fontweight='bold')
    ax.tick_params(colors='#c6d0da')
    for spine in ax.spines.values(): spine.set_color('#46525e')
    ax.grid(alpha=.15,color='#8c9ba9')
    fig.text(.09,.02,'Counts are per frame, NOT unique road defects. Research POC; human verification required.',color='#b8c3cc',fontsize=8)
    fig.tight_layout(rect=(0,.045,1,1))
    fig.savefig(outfile,bbox_inches='tight',facecolor=fig.get_facecolor())
    plt.close(fig)


def process(input_file: Path, output_dir: Path, predictor: Callable,
            max_seconds=0, show=False, keep_mp4v=False, model_tag="Road damage YOLOv8",
            supported_classes=CATEGORY_ORDER):
    if not input_file.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_file}")
    output_dir.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_file}; try MP4/H.264")
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    if not 1 <= fps <= 240:
        fps=25.0
        print("WARNING: Invalid input FPS metadata; assuming 25 fps")
    original_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_seconds > 0:
        limit=max(1,int(max_seconds*fps))
        total=min(total,limit) if total>0 else limit
    if original_w <= 0 or original_h <= 0:
        cap.release()
        raise RuntimeError("Video width/height could not be read")

    print(f"PavePulse AI | {input_file.name} | {original_w}x{original_h} @ {fps:.2f} FPS")
    print(f"Output 1600x900 | {model_tag} | Layout: field inspection / review rail")
    working=output_dir/'pavepulse_working.mp4'
    final=output_dir/'pavepulse_annotated.mp4'
    writer=ensure_writer(working,fps)
    segmenter=ReviewSegmenter(max_gap_frames=max(3,round(fps*0.18)), min_observed_frames=2)
    detections_rows=[]
    timeline=[]
    last_seen=[]
    history=deque(maxlen=250)
    totals={c:0 for c in CATEGORY_ORDER}
    frames_seen={c:0 for c in CATEGORY_ORDER}
    peak_visible=0
    best_frame=None
    best_score=-1
    processed=0
    interrupted=False
    try:
        while True:
            if total>0 and processed>=total: break
            ret,source=cap.read()
            if not ret: break
            dets=predictor(source)
            # Reject unknown classes, empty boxes and bad confidence values (also useful for tests).
            valid=[]
            for det in dets:
                if det.cls in CATEGORY_ORDER and 0.0<=det.confidence<=1.0 and safe_bbox(det.bbox,source.shape[1],source.shape[0]):
                    valid.append(det)
            dets=valid
            seconds=processed/fps
            segmenter.update(processed,dets)
            per_class={c:0 for c in CATEGORY_ORDER}
            for det in dets:
                per_class[det.cls]+=1
                totals[det.cls]+=1
                x1,y1,x2,y2=det.bbox
                detections_rows.append({
                    'frame':processed,'seconds':round(seconds,3),'class':det.cls,
                    'confidence':round(det.confidence,4),'x1':x1,'y1':y1,'x2':x2,'y2':y2,
                })
            for c in CATEGORY_ORDER:
                if per_class[c]>0: frames_seen[c]+=1
            if dets:
                strongest=max(dets,key=lambda d:d.confidence)
                last_seen.append((seconds,strongest.cls,strongest.confidence))
                if len(last_seen)>30: last_seen=last_seen[-30:]
            peak_visible=max(peak_visible,len(dets))
            history.append(len(dets))
            timeline.append({
                'frame':processed,'seconds':round(seconds,3),
                'visible_detections':len(dets),
                **{c:per_class[c] for c in CATEGORY_ORDER},
            })
            dashboard=render_dashboard(
                source,dets,processed,seconds,fps,total,totals,history,last_seen,
                segmenter.completed_count,model_tag,input_file.name,supported_classes,
            )
            score=max((d.confidence for d in dets),default=0.0)
            # First meaningful detected frame is a much better GitHub preview than a random idle frame.
            if best_frame is None or (score>best_score and len(dets)>0):
                best_score=score
                best_frame=dashboard.copy()
            writer.write(dashboard)
            processed+=1
            if processed%max(1,round(fps*2))==0:
                print(f"  {processed} frames ({seconds:.1f}s) | visible proposals={len(dets)} | best confidence={score:.0%}")
            if show:
                try:
                    cv2.imshow("PavePulse AI | Road inspection POC (Q to stop)",cv2.resize(dashboard,(1280,720)))
                    if cv2.waitKey(1)&0xff in (ord('q'),27):
                        interrupted=True
                        break
                except cv2.error:
                    print("WARNING: Preview window unavailable; continuing export without --show")
                    show=False
    finally:
        cap.release()
        writer.release()
        if show:
            try:cv2.destroyAllWindows()
            except cv2.error:pass
    if processed==0:
        working.unlink(missing_ok=True)
        raise RuntimeError("No readable video frames found")
    segments=segmenter.finalize()
    coding="MPEG-4 Part 2 (mp4v)"
    if keep_mp4v:
        working.replace(final)
    else:
        coding=convert_h264(working,final)
    if best_frame is not None:
        cv2.imwrite(str(output_dir/'pavepulse_preview.jpg'),best_frame,[cv2.IMWRITE_JPEG_QUALITY,90])
    write_csv(output_dir/'pavepulse_detections.csv',detections_rows,
              ['frame','seconds','class','confidence','x1','y1','x2','y2'])
    write_csv(output_dir/'pavepulse_timeline.csv',timeline,
              ['frame','seconds','visible_detections',*CATEGORY_ORDER])
    segment_rows=[{
        'segment':idx,'class':s.cls,
        'start_seconds':round(s.start_frame/fps,3),
        'end_seconds':round(s.end_frame/fps,3),
        'observed_frames':s.observed_frames,
        'highest_confidence':round(s.best_confidence,4),
    } for idx,s in enumerate(segments,1)]
    write_csv(output_dir/'pavepulse_review_segments.csv',segment_rows,
              ['segment','class','start_seconds','end_seconds','observed_frames','highest_confidence'])
    create_activity_chart(timeline,output_dir/'pavepulse_activity.png')
    stats=summary_of_timeline(timeline,segments)
    stats.update({
        'project':'PavePulse AI',
        'input_file':input_file.name,
        'source_resolution':f'{original_w}x{original_h}',
        'output_resolution':f'{CANVAS_W}x{CANVAS_H}',
        'fps_source':round(fps,3),
        'seconds_processed':round(processed/fps,3),
        'model':model_tag,
        'model_supported_classes':list(supported_classes),
        'output_codec':coding,
        'ended_early_by_user':interrupted,
        'frame_level_detection_observations_total':len(detections_rows),
        'per_class_detection_frames':frames_seen,
        'per_class_detection_observations':totals,
        'limitations':[
            'Detection counts are observations repeated across frames, not unique defects.',
            'Temporal review segments are class-presence intervals, NOT distinct physical defects.',
            'No measured severity, physical area, accurate GPS, or road health score.',
            'Specialized model may miss defects or generate false positives on out-of-domain footage.',
            'For research demonstration and human review only; not safety-critical operational use.'
        ],
    })
    with open(output_dir/'pavepulse_summary.json','w',encoding='utf-8') as f:
        json.dump(stats,f,indent=2,ensure_ascii=False)
    print(f"\nSUCCESS: Saved {processed} annotated frames to {output_dir.resolve()}")
    for name in ['pavepulse_annotated.mp4','pavepulse_preview.jpg','pavepulse_activity.png',
                 'pavepulse_detections.csv','pavepulse_timeline.csv','pavepulse_review_segments.csv',
                 'pavepulse_summary.json']:
        print('  -',name)
    if not detections_rows:
        print('NOTICE: No road defects detected. Verify video has visible asphalt damage. Try --imgsz 960 or --conf 0.18.')
    else:
        print(f"Observed {len(detections_rows)} frame-level detections; {len(segments)} class review intervals (NOT unique defects).")
    return stats


def main():
    args=read_arguments()
    source=Path(args.input)
    try:
        model=get_detector(args.weights)
    except Exception as exc:
        print(f"MODEL ERROR: {exc}")
        raise SystemExit(1) from exc
    supported_classes = supported_model_classes(model.names)
    if supported_classes == ('pothole',):
        model_tag = 'YOLO | pothole specialist'
    else:
        model_tag = 'YOLOv8s | RDD2022 four-class'

    def predictor(frame):
        if frame.shape[1]>args.analysis_width:
            h,w=frame.shape[:2]
            ratio=args.analysis_width/w
            smaller=cv2.resize(frame,(args.analysis_width,max(1,round(h*ratio))),interpolation=cv2.INTER_AREA)
            detections=infer_detections(model,smaller,args.device,args.imgsz,args.conf)
            scaled=[]
            for d in detections:
                b=safe_bbox([v/ratio for v in d.bbox],w,h)
                if b: scaled.append(Detection(d.cls,d.confidence,b))
            return scaled
        return infer_detections(model,frame,args.device,args.imgsz,args.conf)
    try:
        process(source,Path(args.output),predictor,max_seconds=args.max_seconds,
                show=args.show,keep_mp4v=args.keep_mp4v,model_tag=model_tag,
                supported_classes=supported_classes)
    except (RuntimeError,FileNotFoundError,ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc


if __name__=='__main__':
    main()
