import cv2
import numpy as np
import time
import easyocr

# Path to the pre-trained Haar cascade for car detection and number plate detection
car_cascade_src = '/Users/anand/Downloads/cars1.xml'  # Update this path if necessary
plate_cascade_src = '/Users/anand/Downloads/haarcascade_russian_plate_number (1).xml'  # Update this path if necessary
video_src = r'/Users/anand/Downloads/h1.mp4'  # Update this path if necessary

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # Add more languages if needed

# Initialize car number for tracking
car_count = 0

# Define color ranges in HSV (Hue, Saturation, Value)
COLOR_RANGES = {
    'Red': ([0, 100, 100], [10, 255, 255]),
    'Green': ([40, 100, 100], [80, 255, 255]),
    'Blue': ([100, 100, 100], [140, 255, 255]),
    'Yellow': ([20, 100, 100], [35, 255, 255]),
    'White': ([0, 0, 200], [180, 30, 255]),
    'Black': ([0, 0, 0], [180, 255, 30]),
}

def detect_color(hsv_roi):
    color_found = "Unknown"
    max_percentage = 0
    for color_name, (lower_bound, upper_bound) in COLOR_RANGES.items():
        lower_bound = np.array(lower_bound, dtype="uint8")
        upper_bound = np.array(upper_bound, dtype="uint8")
        mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)
        color_percentage = (cv2.countNonZero(mask) / (hsv_roi.size / 3)) * 100
        if color_percentage > max_percentage:
            max_percentage = color_percentage
            color_found = color_name
    return color_found if max_percentage > 20 else "Unknown"

def calculate_speed(pos1, pos2, time_diff, ppm):
    """Calculate speed based on pixel movement and time difference."""
    pixel_distance = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
    meters = pixel_distance / ppm
    speed = (meters / time_diff) * 3.6  # Convert m/s to km/h
    return speed

# Load the pre-trained Haar cascade for car detection and number plate detection
car_cascade = cv2.CascadeClassifier(car_cascade_src)
plate_cascade = cv2.CascadeClassifier(plate_cascade_src)

# Capture video from the video source
cap = cv2.VideoCapture(video_src)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Assume 4 meters per 100 pixels (adjust as needed for your video)
pixels_per_meter = 25

# Dictionary to store car tracking info
car_tracks = {}

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 3, minSize=(80, 80))
    
    current_time = time.time()
    
    for (x, y, w, h) in cars:
        car_center = (int(x + w / 2), int(y + h / 2))
        
        # Check if this car is already being tracked
        tracked = False
        tracked_id = None
        for car_id, car_info in list(car_tracks.items()):
            dist = np.sqrt((car_center[0] - car_info['center'][0])**2 + 
                           (car_center[1] - car_info['center'][1])**2)
            if dist < 50:  # Assume it's the same car if the center is within 50 pixels
                tracked = True
                tracked_id = car_id
                if current_time - car_info['time'] > 0.1:  # Update every 0.1 seconds
                    speed = calculate_speed(car_info['center'], car_center, 
                                            current_time - car_info['time'], 
                                            pixels_per_meter)
                    car_tracks[car_id] = {
                        'center': car_center,
                        'time': current_time,
                        'speed': speed
                    }
                break
        
        if not tracked:
            car_count += 1
            tracked_id = car_count
            car_tracks[tracked_id] = {
                'center': car_center,
                'time': current_time,
                'speed': 0
            }
        
        # Draw rectangle around the car
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Detect color
        car_roi = frame[y:y + h, x:x + w]
        hsv_roi = cv2.cvtColor(car_roi, cv2.COLOR_BGR2HSV)
        detected_color = detect_color(hsv_roi)
        
        # Display color and speed information
        cv2.putText(frame, f"Color: {detected_color}", (x, y - 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"Speed: {car_tracks[tracked_id]['speed']:.2f} km/h", 
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Detect the number plate
        plate_roi = frame[y:y + h, x:x + w]
        plate_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(plate_gray, 1.1, 3, minSize=(30, 10))
        
        for (px, py, pw, ph) in plates:
            cv2.rectangle(frame, (x + px, y + py), (x + px + pw, y + py + ph), (0, 255, 0), 2)
            # Crop and OCR the number plate
            plate_image = plate_roi[py:py + ph, px:px + pw]
            plate_text = reader.readtext(plate_image)
            if plate_text:
                # Extract the text from the result
                plate_text_str = " ".join([text[1] for text in plate_text]).strip()
                cv2.putText(frame, plate_text_str, (x + px, y + py - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Slow down the video by adjusting the playback speed
    time.sleep(1 / fps)

    # Display the frame
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
