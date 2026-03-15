import argparse
import cv2


def parse_args():
    p = argparse.ArgumentParser(description="Step 1: OpenCV video/webcam preview")
    p.add_argument(
        "--source",
        required=True,
        help="0 for webcam, or path to a video file",
    )
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    return p.parse_args()


def main():
    args = parse_args()

    # OpenCV expects int for webcam index, string path for file
    source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    # Try to set resolution (works mainly for webcams)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    while True:
        ok, frame = cap.read()
        if not ok:
            # End of file or camera read error
            break

        cv2.imshow("TrashDetectionSystem - Step 1 Preview (press q)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()