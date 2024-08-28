import cv2,os,sys,argparse,shutil

def main(video_path, label_dir, save_dir):
    valid_idx = [int(label[:-4].split('frame')[1]) for label in os.listdir(label_dir)]
    min_idx = min(valid_idx)

    cap = cv2.VideoCapture(video_path)
    frame_idx = min_idx
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or count == len(valid_idx):
            cap.release()
            break
        if frame_idx in valid_idx:
            cv2.imwrite(os.path.join(save_dir, f'frame{frame_idx:05d}.png'), frame)
            count += 1
            print(f'\rsaving {count}...',end='')

        frame_idx += 1

if __name__ == '__main__':
    pwd = sys.path[0]

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--video_path', type=str, default=os.path.join(pwd, 'pick_video.mp4'))
    parser.add_argument('-l', '--label_dir', type=str, default=os.path.join(pwd, 'labels'))
    parser.add_argument('-s', '--save_dir', type=str, default=os.path.join(pwd, 'pick_img'))
    args = parser.parse_args()
    args.video_path = os.path.abspath(args.video_path)
    args.label_dir = os.path.abspath(args.label_dir)
    args.save_dir = os.path.abspath(args.save_dir)
    assert os.path.exists(args.video_path), 'video not found'
    assert os.path.exists(args.label_dir), 'label dir not found'
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)

    main(video_path=args.video_path, 
         label_dir=args.label_dir, 
         save_dir=args.save_dir)