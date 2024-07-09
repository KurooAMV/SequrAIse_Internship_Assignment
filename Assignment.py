import cv2
import numpy as np
import pandas as pd
import time
from datetime import timedelta

def timeInFormat(startTime,endTime):
    hours, rem = divmod(endTime-startTime, 3600)
    minutes, seconds = divmod(rem, 60)
    return ("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

# Define video capture
video_path = 'C:/Users/stuti/Downloads/AI Assignment video.mp4'  # Replace with the actual video path
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# print(frame_height,frame_width,total_frames,fps,sep=",")

# Define color ranges for HSV


# precise_color_ranges = {
# 'Yellow': ([40, 50, 20], [55, 60, 80]), 
# 'Blue': ([160, 40, 40],[180, 60, 40]), 
# 'White': ([40, 15, 35],[70, 10, 110]), 
# 'Orange': ([5, 60, 55], [10, 50, 100])
# }

color_ranges = {
    'Yellow': ([200, 200, 0], [255, 255, 100]),
    'Blue': ([100, 0, 0], [255, 100, 100]),
    'White': ([200, 200, 200], [255, 255, 255]),
    'Orange': ([0, 100, 200], [100, 180, 255])
}

# Define quadrants
quadrants = {
'1': ((1230,519), (1753,994)),
'2': ((774,510), (1235,1016)),
'3': ((774,1), (1230,510)),
'4': ((1245,1), (1756,507))
}

# print(quadrants)
# print(color_ranges)
# Initialize tracking data and event log
tracking_data = {}
event_log = []

def get_quadrant(cx, cy, quadrants):
    for quadrant, ((x1, y1), (x2, y2)) in quadrants.items():
        # print(f"x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}, quadrant")
        if x1 <= cx < x2 and y1 <= cy < y2:
            return quadrant
    return None

# Initialize video writer for processed video
fourcc = cv2.VideoWriter.fourcc(*'mp4v')
out = cv2.VideoWriter('processed_video.mp4', fourcc, fps, (frame_width, frame_height))
print("Video processing....")

startTime = time.time()
# Process the video
frame_number = 0
# print(tracking_data)
while cap.isOpened():
    ret, frame = cap.read()
    # print(ret)
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # cv2.imshow('HSV Scale', hsv) 
    # cv2.waitKey(0) 
    
    for color_name, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        # mask = cv2.inRange(frame,lower,upper)
        kernel = np.ones((3,3), np.uint8)
        # cv2.imshow("mask",mask)
        # cv2.waitKey(1)
        # Apply erosion and dilation to remove and smal disturbance in the image
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)


        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    
        for contour in contours:
            # cv2.imshow("contour",contour)
            # cv2.waitKey(1)
            ((cx, cy), radius) = cv2.minEnclosingCircle(contour)
            if radius > 10:
                cx,cy,radius=int(cx),int(cy),int(radius)
                # print(f"center: ({cx}, {cy}), radius: {radius}")
                cv2.circle(contour,(cx,cy),radius,(0,255,0),2)
                
                # print(f"Processing color: {color_name}, center: ({cx}, {cy}), radius: {radius}\n\n")
                current_quadrant = get_quadrant(cx, cy, quadrants)
                # print(f"\nCurrent quadrant: {current_quadrant}")

                if (color_name, cx, cy) not in tracking_data:
                    tracking_data[(color_name, cx, cy)] = None
                
                prevTime = time.time()
                previous_quadrant = tracking_data[(color_name, cx, cy)]
                # print(tracking_data)
                # print(previous_quadrant)

                if previous_quadrant != current_quadrant:
                    
                    # print(f"\nprevious quad: {previous_quadrant}\ncurrent quad : {current_quadrant}.")
                    timestamp = time.time()
                    # print(f"\ntimestamp: {timestamp}")
                    
                    if current_quadrant is not None:
                        event_log.append((timestamp, current_quadrant, color_name, 'Entry'))
                        cv2.putText(frame, f"Entry at {timeInFormat(prevTime,timestamp)}", (cx - radius, cy - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        # print(f"\nEntry logged: {timestamp:.1f}s, {current_quadrant}, {color_name}")
                    
                    if previous_quadrant is not None:
                        event_log.append((timestamp, previous_quadrant, color_name, 'Exit'))
                        cv2.putText(frame, f"Exit at {timeInFormat(prevTime,timestamp)}", (cx - radius, cy + radius + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        # print(f"\nExit logged: {timestamp:.1f}s, {previous_quadrant}, {color_name}")

                    tracking_data[(color_name, cx, cy)] = current_quadrant
                    # print(tracking_data)

                # Draw the detected circle
                cv2.circle(frame, (cx, cy), radius, (0, 255, 0), 3)
                cv2.putText(frame, color_name, (cx - radius, cy - radius - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)

    frame_number += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # print("Wait_key evaluated correctly!")
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Saving event log to text file
if event_log:
    event_df = pd.DataFrame(event_log, columns=['Time', 'Quadrant Number', 'Ball Colour', 'Event Type'])
    event_df.to_csv('event_log.txt', index=False, sep=',', header=True)
    print("Event log saved to 'event_log.txt'.")
else:
    print("No events detected.")

print("Processing complete. Check 'processed_video.mp4' and 'event_log.txt'.")
endTime = time.time()
print(f"Total TIme taken: {timeInFormat(startTime,endTime)}")
