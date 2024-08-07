import cv2,os,sys

def main(video_path, label_dir, save_dir):
    valid_idx = list(map(lambda x: int(x[:-4].split('frame')[1]), os.listdir(label_dir)))

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            break
        if frame_idx in valid_idx:
            cv2.imwrite(os.path.join(save_dir, f'frame{frame_idx:05d}.png'), frame)
            count += 1
            print(f'\rsaving {count}...',end='')

        frame_idx += 1

if __name__ == '__main__':
    pwd = sys.path[0]
    label_dir = os.path.join(pwd, 'labels')
    save_dir = os.path.join(pwd, 'pick_img')
    video_path = os.path.join(pwd, 'pick_video.mp4')

    main(video_path, label_dir, save_dir)