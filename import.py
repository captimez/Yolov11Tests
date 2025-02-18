import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

class DepthCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(config)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return False, None, None
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return True, depth_image, color_image

# Initialer Punkt für Tiefenmessung
point = (300, 400)

def show_distance(event, x, y, flags, param):
    """Speichert die Koordinaten des Mausklicks für die Tiefenmessung"""
    global point 
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)

# YOLO Modell laden
model = YOLO(".yolov11-seg/my_model.pt")

# Kamera initialisieren
dc = DepthCamera()

# Fenster erstellen
cv2.namedWindow('Depth Frame')
cv2.namedWindow('Color Frame')

# Maus-Callback setzen
cv2.setMouseCallback('Color Frame', show_distance)

while True:
    ret, depth_frame, color_frame = dc.get_frame()
    if not ret:
        continue

    # YOLO-Inferenz auf das Farbbild
    results = model(color_frame)

    for result in results:
        boxes = result.boxes.xyxy  # Bounding Boxes
        confs = result.boxes.conf  # Confidence Scores
        masks = result.masks.data  # Segmentation Masks 
        classes = result.boxes.cls  # Class IDs

        masks_converted = []
        for mask in masks:
            mask = mask.cpu().numpy()  # Convert to NumPy array
            mask = cv2.resize(mask, (color_frame.shape[1], color_frame.shape[0]))  # Resize mask to match image
            mask = (mask * 255).astype(np.uint8)  # Convert to 8-bit image

            # Apply color map to make the mask visible
            mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)  

            masks_converted.append(mask)

        # Iterate over detected objects & overlay masks
        for box, conf, cls, mask in zip(boxes, confs, classes, masks_converted):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"

            # Compute the center of the object
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            # Get depth information at the object's center
            if 0 <= center_x < depth_frame.shape[1] and 0 <= center_y < depth_frame.shape[0]:
                distance_z = depth_frame[center_y, center_x]
            else:
                distance_z = 0  # Out of bounds

            # Draw bounding box
            cv2.rectangle(color_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(color_frame, (center_x, center_y), 4, (0, 0, 255), -1)
            cv2.putText(color_frame, f"{label} {distance_z}mm", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Blend mask onto color frame
            color_frame = cv2.addWeighted(color_frame, 0.7, mask, 0.3, 0)  # ✅ Corrected blending


    # Kreis an der Klick-Position zeichnen & Tiefenwert anzeigen
    cv2.circle(color_frame, point, 4, (255, 0, 0), -1)
    if 0 <= point[0] < depth_frame.shape[1] and 0 <= point[1] < depth_frame.shape[0]:
        distance_z = depth_frame[point[1], point[0]]
        cv2.putText(color_frame, f'{distance_z} mm', (point[0], point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Tiefenbild einfärben
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.2), cv2.COLORMAP_JET)

    # Bilder anzeigen
    cv2.imshow('Depth Frame', depth_frame)
    cv2.imshow('Color Frame', color_frame)
    cv2.imshow('Depth Map', depth_colormap)

    # Beenden mit 'ESC'
    key = cv2.waitKey(1)
    if key == 27:
        cv2.destroyAllWindows()
        break

