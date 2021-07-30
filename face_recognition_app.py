import face_recognition  # package face-recognition
import cv2  # package opencv-python

webcam = cv2.VideoCapture(0)

image_file = input("Target The first Image File > ")
target_image = face_recognition.load_image_file(image_file)
target_encoding = face_recognition.face_encodings(target_image)[0]

# second_image_file = input("Target The second Image File > ")
# second_target_image = face_recognition.load_image_file(second_image_file)
# second_target_encoding = face_recognition.face_encodings(second_target_image)[0]

print("Image Loaded 128-dimension. Face encoding generated \n")

target_name = input("Target Name > ")

# result = face_recognition.compare_faces(
#     [target_encoding], second_target_encoding, tolerance=0.6
# )

# print(result)

process_this_frame = True

while True:
    ret, frame = webcam.read()

    small_frame = cv2.resize(frame, None, fx=0.2, fy=0.2)
    rgb_small_frame = cv2.cvtColor(small_frame, 4)

    if process_this_frame:
        face_location = face_recognition.face_locations(rgb_small_frame)
        frame_encodings = face_recognition.face_encodings(rgb_small_frame)

        if frame_encodings:
            frame_face_encoding = frame_encodings[0]
            match = face_recognition.compare_faces(
                [frame_face_encoding], target_encoding, tolerance=0.5
            )
            label = target_name if match[0] else "Unknown"

    process_this_frame = not process_this_frame

    if face_location:
        top, right, bottom, left = face_location[0]

        top *= 5
        right *= 5
        bottom *= 5
        left *= 5
        color = (0, 255, 0) if match[0] else (0, 0, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
        label_font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(
            frame, label, (left + 6, bottom - 10), label_font, 0.5, (0, 0, 0), 1
        )

    cv2.imshow("Video Feed ", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()
