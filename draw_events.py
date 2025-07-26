
import os
if os.name == 'nt':
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import yaml

class MouseStatus:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.lb = False
        self.rb = False
        self.lb_clk = False
        self.rb_clk = False

def mouse_event_handler(event, x, y, flags, param):
    global mouse_status
    mouse_status.x = x
    mouse_status.y = y
    match event:
        case cv2.EVENT_LBUTTONDOWN:
            mouse_status.lb = True
        case cv2.EVENT_RBUTTONDOWN:
            mouse_status.rb = True
        case cv2.EVENT_LBUTTONUP:
            mouse_status.lb_clk = True
            mouse_status.lb = False
        case cv2.EVENT_RBUTTONUP:
            mouse_status.rb_clk = True
            mouse_status.rb = False

def draw(img, results):
    for result in results:
        if result['type'] == 'wire':
            points = result['points']
            if len(points) == 2:
                cv2.line(img, tuple(points[0]), tuple(points[1]), (0, 255, 0), 2)
        elif result['type'] == 'area':
            contour = np.array(result['points'], dtype=np.int32)
            cv2.polylines(img, [contour], True, (255, 0, 0), 2)
            cv2.putText(img, 'Area', tuple(contour[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

def usage():
    print("Usage: python draw_events.py")
    print("Left click to add points, right click to finalize area or wire segment.")
    print("Press Backspace to remove the last point.")
    print("Press ESC to exit and save the configuration to config.yaml.")


def main():
    global mouse_status
    mouse_status = MouseStatus()
    window_name = 'Frame'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_event_handler)

    usage()

    cap = None
    key = 0
    points = []
    items = []

    video_source = 'people-detection.264'
    #video_source = 0  # Default to the first camera

    while key != 27:  # 27 is the ESC key
        if cap is None:
            cap = cv2.VideoCapture(video_source)
        ret, frame = cap.read()
        if not ret or frame is None:
            cap = None
            continue
        
        out_img = frame.copy()

        if mouse_status.lb_clk:
            mouse_status.lb_clk = False
            points.append([mouse_status.x, mouse_status.y])
            print(f"Point added: {mouse_status.x}, {mouse_status.y}")

        if mouse_status.rb_clk:
            mouse_status.rb_clk = False
            if len(points) == 2:
                obj = {'type': 'wire', 'points': points.copy()}
                items.append(obj)
                print(f"Wire segment added: {points[:2]}")
                points = []
            if len(points) >= 2:
                obj = {'type': 'area', 'points': points.copy()}
                items.append(obj)
                area_contour = np.array(points, dtype=np.int32)
                print(f"Area contour added: {area_contour.tolist()}")
                points = []

        if len(points) > 0:
            contour = [[mouse_status.x, mouse_status.y]]
            contour = contour + points
            contour = np.array(contour, dtype=np.int32)
            cv2.polylines(out_img, [contour], True, (255, 0, 0), 2)

        draw(out_img, items)

        cv2.imshow(window_name, out_img)
        key = cv2.waitKey(1)

        if key == 8:  # Backspace key
            if points:
                points = points[:-1]

    cap.release()
    cv2.destroyAllWindows()

    result = yaml.dump(items, allow_unicode=True, default_flow_style=False)
    print(result)

    with open('config.yaml', 'w') as file:
        file.write(result)



if __name__ == "__main__":
    main()
