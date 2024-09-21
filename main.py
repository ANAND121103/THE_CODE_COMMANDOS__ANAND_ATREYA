import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("/Users/anand/Downloads/yolov3.weights", "/Users/anand/Downloads/yolov3.cfg")

# Get the names of all layers in the network
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers().flatten()
output_layers = [layer_names[i - 1] for i in output_layers_indices]

# Load COCO class labels
with open("/Users/anand/Downloads/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Predefined list of basic colors for detection
COLORS = {
    "Red": (255, 0, 0),
    "Green": (0, 255, 0),
    "Blue": (0, 0, 255),
    "Yellow": (255, 255, 0),
    "Cyan": (0, 255, 255),
    "Magenta": (255, 0, 255),
    "White": (255, 255, 255),
    "Gray": (128, 128, 128),
    "Black": (0, 0, 0)
}

# Function to get color name from BGR value
def get_color_name(bgr_color):
    rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
    tolerance = 60
    def within_tolerance(c1, c2):
        return all(abs(c1[i] - c2[i]) <= tolerance for i in range(3))
    if abs(rgb_color[0] - rgb_color[1]) < 20 and abs(rgb_color[1] - rgb_color[2]) < 20:
        return "Gray"
    closest_color_name = None
    min_distance = float('inf')
    for color_name, color_value in COLORS.items():
        if within_tolerance(rgb_color, color_value):
            distance = np.linalg.norm(np.array(rgb_color) - np.array(color_value))
            if distance < min_distance:
                min_distance = distance
                closest_color_name = color_name
    return closest_color_name if closest_color_name else "Unknown"

# Speed calculation helper function
def calculate_speed(prev_frame, curr_frame, fps):
    distance = np.linalg.norm(np.array(curr_frame) - np.array(prev_frame))
    speed_mps = (distance * fps) / 30.0
    speed_kmph = speed_mps * 3.6
    return speed_kmph

# License plate detection using contours
def detect_license_plate(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 100, 200)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    license_plate = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h
            if 2 < aspect_ratio < 5:
                license_plate = frame[y:y+h, x:x+w]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                break
    
    return license_plate

# Placeholder function for character recognition
def recognize_characters(license_plate_image):
    # Implement character recognition here using a suitable model
    return "ABC123"

# Function to process a video file and detect vehicles (cars, trucks, motorbikes) and license plates
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to open video file.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    prev_frame_vehicle_positions = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        height, width, _ = frame.shape
        
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        vehicle_positions = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] in ["car", "motorbike", "truck"]:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    vehicle_positions.append((center_x, center_y))

                    vehicle_type = classes[class_id]
                    vehicle_roi = frame[y:y+h, x:x+w]
                    avg_color = np.mean(vehicle_roi, axis=(0, 1)).astype(int)
                    color_name = get_color_name(avg_color)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{vehicle_type.capitalize()}, Color: {color_name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    license_plate = detect_license_plate(frame)
                    if license_plate is not None:
                        registration_number = recognize_characters(license_plate)
                        cv2.putText(frame, f"Plate: {registration_number}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if prev_frame_vehicle_positions:
            for i, curr_pos in enumerate(vehicle_positions):
                if i < len(prev_frame_vehicle_positions):
                    speed_kmph = calculate_speed(prev_frame_vehicle_positions[i], curr_pos, fps)
                    cv2.putText(frame, f"Speed: {speed_kmph:.2f} km/h", (curr_pos[0], curr_pos[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        prev_frame_vehicle_positions = vehicle_positions

        cv2.imshow("Vehicle Detection and License Plate Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Tkinter GUI for uploading and processing the video
def upload_video():
    video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
    if video_path:
        process_video(video_path)

# Enhanced Tkinter GUI
def create_gui():
    root = tk.Tk()
    root.title("PAriVahaN - Vehicle Detection App")
    root.geometry("500x400")
    root.configure(bg="#2C3E50")

    # Header label
    header = tk.Label(root, text="Welcome to PAriVahaN", font=("Arial", 24, "bold"), fg="#ECF0F1", bg="#2C3E50")
    header.pack(pady=20)

    # Instruction label
    instructions = tk.Label(root, text="Upload a video to detect vehicles, their speed, and license plates", font=("Arial", 14), fg="#ECF0F1", bg="#2C3E50")
    instructions.pack(pady=10)

    # Upload button with styling
    upload_button = tk.Button(root, text="Upload Video", command=upload_video, font=("Arial", 16), bg="#3498DB", fg="#FFFFFF", width=20)
    upload_button.pack(pady=20)

    # Footer label
    footer = tk.Label(root, text="Powered by PAriVahaN", font=("Arial", 12), fg="#BDC3C7", bg="#2C3E50")
    footer.pack(side=tk.BOTTOM, pady=10)

    root.mainloop()

# Run the enhanced GUI
create_gui()
