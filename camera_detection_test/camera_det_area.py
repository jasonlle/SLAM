import cv2
import pandas
from datetime import datetime

static_back = None
motion_list = [None, None]
times = []

video = cv2.VideoCapture(0)

# --- Define the red detection box (x1, y1) top-left and (x2, y2) bottom-right ---
roi_x1, roi_y1 = 100, 100
roi_x2, roi_y2 = 400, 400

while True:
    check, frame = video.read()
    if not check:
        break

    motion = 0
    inside_roi = 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if static_back is None:
        static_back = gray
        continue

    diff_frame = cv2.absdiff(static_back, gray)
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    cnts, _ = cv2.findContours(
        thresh_frame.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue

        motion = 1
        (x, y, w, h) = cv2.boundingRect(contour)

        # Draw green box around detected motion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Check if detected motion overlaps the red ROI
        if (x < roi_x2 and x + w > roi_x1 and y < roi_y2 and y + h > roi_y1):
            inside_roi = 1

    # Draw the red ROI box
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)

    # Show text depending on whether motion is inside the ROI
    if inside_roi:
        cv2.putText(frame, "INSIDE AREA", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "OUTSIDE AREA", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    motion_list.append(inside_roi)
    motion_list = motion_list[-2:]

    if motion_list[-1] == 1 and motion_list[-2] == 0:
        times.append(datetime.now())

    if motion_list[-1] == 0 and motion_list[-2] == 1:
        times.append(datetime.now())

    cv2.imshow("Gray Frame", gray)
    cv2.imshow("Difference Frame", diff_frame)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        if inside_roi == 1:
            times.append(datetime.now())
        break

rows = []
for i in range(0, len(times), 2):
    if i + 1 < len(times):
        rows.append({"Start": times[i], "End": times[i + 1]})

df = pandas.DataFrame(rows)
df.to_csv("Time_of_movements.csv", index=False)

video.release()
cv2.destroyAllWindows()