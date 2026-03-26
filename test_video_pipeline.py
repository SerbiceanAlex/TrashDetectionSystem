"""Quick test: can we run inference on a video frame without crashing?"""
import sys
import traceback

try:
    from app.inference import load_models, run_pipeline_frame
    import cv2
    import numpy as np

    print("Loading models...", flush=True)
    load_models()
    print("Models loaded OK", flush=True)

    print("Creating test frame...", flush=True)
    frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    print(f"Frame shape: {frame.shape}", flush=True)

    print("Running pipeline on frame...", flush=True)
    detections, annotated, ms = run_pipeline_frame(frame)
    print(f"DONE: {len(detections)} detections, {ms:.1f}ms, annotated={annotated.shape}", flush=True)

    # Now test with VideoCapture
    print("\nTesting with VideoCapture...", flush=True)
    cap = cv2.VideoCapture("test_video.mp4")
    ret, vframe = cap.read()
    cap.release()
    if ret:
        print(f"Video frame: {vframe.shape}", flush=True)
        detections2, annotated2, ms2 = run_pipeline_frame(vframe)
        print(f"Video DONE: {len(detections2)} detections, {ms2:.1f}ms", flush=True)
    else:
        print("ERROR: Could not read video frame", flush=True)

    # Test VideoWriter
    print("\nTesting VideoWriter...", flush=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("test_output.mp4", fourcc, 10, (320, 240))
    writer.write(annotated if annotated.shape[1] == 320 else cv2.resize(annotated, (320,240)))
    writer.release()
    print("VideoWriter OK", flush=True)

    print("\nALL TESTS PASSED", flush=True)
except Exception:
    traceback.print_exc()
    sys.exit(1)
