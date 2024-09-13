import os,argparse,sys,traceback,time,subprocess
import rich,cv2
import numpy as np
from queue import Queue
from rich.text import Text
from rich.panel import Panel
from functools import partial
from signal import signal, SIGINT
from threading import Thread, Event
from ultralytics.utils import yaml_load

from reactor import Reactor
from model import Yolo8_Detracker
from utils import DataSource, LoadRsCamera

def update_args(args):
    config = yaml_load(args.config) # dict
    for key, value in config.items():
        setattr(args, key, value) 
    return args

def terminate_handle(sig, frame, 
                     producer_thread, consumer_threads, 
                     queues, terminator):
    print('\nSignal received, waiting for the completion of current batch ...')
    
    terminator.set()

    producer_thread.join() # waiting for yolo8 model to finish the last batch
    
    for queue,thread in zip(queues,consumer_threads):
        if queues is not None:
            queue.put("EOP") # send end-of-process signal to consumer thread
            thread.join()
    
    print('Exiting.')
    sys.exit(0)

def main(args, terminator, res_queue=None, alert_queue=None):
    try:
        datas = DataSource(args)
        args.if_track = getattr(args, "enable_track", \
                False if os.path.exists(args.input) and datas.data_iter.nf == datas.data_iter.ni else True) # 0 for only imgs

        model = Yolo8_Detracker(args)
        datas.data_iter.bs = getattr(args,"batch",1) # synchronise data_iter.bs with model batch size

        print('✊ WARMING UP...')
        model.warmup() # warm up inferencer
        
        reactor = Reactor(args)
        with model._lock: # for thread-safe inference
            for _ in range(2): # ring twice meaning detection begins
                os.system('play --no-show-progress --null --channels 1 synth 0.1 sine 1000')
            
            for path_bs, img_bs, info_bs in datas: # *_bs are list with a len of args.batch
                if terminator.is_set(): # terminate the program when encounter KeyBoardInterrupt
                    break
                
                for res_id, res in enumerate(model(path_bs, img_bs)):  # batch_res is a list of Frame_Info objs
                    frame_emergent_data = res.emergency_info # ndarray, (num_boxes, 4), [cls, track_id, light_color, depth]
                    v0, brake_a, if_alert, beep_info = reactor.update(frame_emergent_data, res.fps) # unit: km/h, m/s^2
                    
                    res.v0 = v0 # unit: km/h
                    res.brake_a = brake_a
                    res.if_alert = if_alert

                    if alert_queue is not None:
                        alert_queue.put(beep_info)

                    if res_queue is not None:
                        res_queue.put(res)
                    
                    if args.verbose:
                        info = [info_bs[res_id], res.verbose(), ' ⚠' if if_alert else '']
                        rich.print(f"[bold][blue]{info[0]}[/blue][green]{info[1]}[/green][red]{info[2]}[/red][/bold]")
                        if res_queue is not None:
                            res_queue.put(''.join(info).strip())
                    
                    if args.show:
                        res.plot(show=True)
                    
    except:
        print(traceback.format_exc())
    
    finally:
        cv2.destroyAllWindows()

        if isinstance(datas.data_iter,LoadRsCamera):
            datas.data_iter.pipeline.stop()
                
        for queue in (alert_queue, res_queue):
            if queue is not None:
                queue.put('EOP') # end of process
        
        for _ in range(2): # ring twice meaning program ends
                os.system('play --no-show-progress --null --channels 1 synth 0.1 sine 1000')
        # for v in self.vid_writer.values():
        #     if isinstance(v, cv2.VideoWriter):
        #         v.release()

def alert(alert_queue):
    if_speaker = subprocess.getoutput("aplay -l | grep -i usb") # check if speaker is connected
    while True:
        beep_info = alert_queue.get() # (duration, frequency) or None or 'EOP'
        if beep_info == 'EOP': # end of process
            break

        if beep_info is not None and if_speaker:
            duration, frequency = beep_info
            if duration*frequency != 0:
                os.system(f'play --no-show-progress --null --channels 1 synth {duration} sine {frequency}')

def save_results(args, res_queue):
    count = 0
    img_save_path = os.path.join(args.save_dir, 'img')
    meta_save_path = os.path.join(args.save_dir, 'meta')
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(meta_save_path, exist_ok=True)

    img_saving_mode = getattr(args, "save_img_format", 'gray').lower()

    with open(os.path.join(args.save_dir, 'terminal_output.txt'), 'w') as info_writer:
        while True:
            res = res_queue.get()
            if res == 'EOP': # end of process
                break
            
            if isinstance(res, str): # info
                info_writer.write(res+'\n')
            else:   # Frame_Info obj saved as pkl
                save_name = os.path.splitext(os.path.basename(res.path))[0]
                if not args.input.isnumeric():
                    save_name = f'{save_name}_{count}'

                img, meta_data = res.compact_data
                cv2.imwrite(os.path.join(img_save_path, save_name+'.png'), img if img_saving_mode == 'rgb' else img[...,0])
                np.save(os.path.join(meta_save_path, save_name+'.npy'), meta_data)
            
                count += 1 
                
    print(f'Capture {count} frames in total.')

if __name__ == '__main__':
    os.system('play --no-show-progress --null --channels 1 synth 0.1 sine 2000')

    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model', type=str, default='../Weights/yolo8s_half_sim_288x480.engine', help='model path')
    parser.add_argument('-c','--config',type=str, default=os.path.join(sys.path[0],'TZP.yaml'), help='config file path')
    parser.add_argument('-i','--input',type=str, default='0', help='input path, can be imgs, videos, directories, URLs or int for webcam')
    parser.add_argument('-s','--save_dir',type=str, default='', help='frame result save path')
    parser.add_argument('--show',action='store_true', default=False, help='show frame detection results on screen')
    parser.add_argument('--verbose',action='store_true', default=False, help='print frame detection results in terminal')
    args = parser.parse_args()
    args.model = os.path.abspath(args.model)
    args.config = os.path.abspath(args.config)
    args.input = os.path.abspath(args.input) if not args.input.isnumeric() else args.input
    args.save_dir = os.path.abspath(args.save_dir) if args.save_dir else ''
    assert os.path.exists(args.model), f'Model file of which path is {args.model} doesn\'t exist'
    assert os.path.exists(args.config), f'Config file of which path is {args.config} doesn\'t exist'
    assert args.config.endswith(('yaml','yml')), f'Config file must be a yaml file, but got {args.config}'
    
    args = update_args(args)

    if args.save_dir:
        run_time = str(time.strftime('%Y-%m-%dT%H-%M-%S', time.localtime()))
        args.save_dir = os.path.join(args.save_dir, run_time)
        os.makedirs(args.save_dir, exist_ok=True)
        
    res_queue = Queue() if args.save_dir else None
    alert_queue = Queue()
    terminator = Event()

    main_thread = Thread(target=main, args=(args, terminator, res_queue, alert_queue))
    save_thread = Thread(target=save_results, args=(args, res_queue)) if args.save_dir else None
    alert_thread = Thread(target=alert, args=(alert_queue,))
    signal(SIGINT, partial(terminate_handle,  # capture KILL (Ctrl+C) signal and stop the program
                           producer_thread=main_thread, 
                           consumer_threads=[save_thread,alert_thread] if args.save_dir else [alert_thread],
                           queues=[res_queue,alert_queue] if args.save_dir else [alert_queue], # seqence should be the same as consumer_threads
                           terminator=terminator))
    
    rich.print(Panel(Text("Press Ctrl+C to stop the program!", justify="center", style='bold green italic'), 
                     title="Note",
                     box=rich.box.DOUBLE,
                     border_style='green'))

    main_thread.start()
    alert_thread.start()
    if args.save_dir:
        save_thread.start()

    main_thread.join()
    alert_thread.join()
    if args.save_dir:
        save_thread.join()



