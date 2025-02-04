import cv2
import os

def video_to_frames(video_path, output_folder):

    video_capture = cv2.VideoCapture(video_path)
    

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    
   
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_count = 0
    

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        

        frame_filename = os.path.join(output_folder, f"{frame_count:04d}.jpg")
        

        cv2.imwrite(frame_filename, frame)
        

        print(f"加工 {frame_count + 1}/{total_frames}")
        
        frame_count += 1
    

    video_capture.release()
    print("視訊處理完成!")
for i in range (1,9):
    video_path = f"{i}.mp4" 
    output_folder = f'output{i}'
    video_to_frames(video_path, output_folder)
