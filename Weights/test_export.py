import os, argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model', type=str, required=True, help='pytorch model path')
parser.add_argument('-s','--source',type=str, required=True, default='/home/yzq/hzy/yolov8/demo', help='source path, can be imgs, videos, directories, URLs or int for webcam')
parser.add_argument('-d','--save_path',type=str, default='', help='result save dir')
parser.add_argument('-c','--config',type=str, default='/home/yzq/hzy/yolov8/cfg/TZP.yaml', help='config file path')
parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
parser.add_argument('--iou', type=float, default=0.45, help='iou threshold')
args = parser.parse_args()
assert os.path.exists(args.model), f'Model of which path is {args.pt_model} doesn\'t exist'
if not args.source.isnumeric(): 
    assert os.path.exists(args.source), f'Input path({args.pt_model}) doesn\'t exist'

# Load a model
model = YOLO(args.model, 
             task='detect',
             verbose=True)  

# Custom inference args
custom = {"conf": args.conf,
          "iou": args.iou,
          "mode": "predict",
          "data": args.config,
          "half":True if 'half' in args.model else False,
          "dnn":True if 'onnx' in os.path.splitext(args.model)[-1] else False}  # method defaults

model.predictor = model._smart_load("predictor")(overrides=custom, _callbacks=model.callbacks)
model.predictor.setup_model(model=model.model, verbose=False)

# Run inference, return a list of Results objects
## argument conform default.yaml
results = model(args.source)

## `stage_times` is a list with len of instances' num, 
## each item has a speed attribute that record each stage time cost
stage_times = model.predictor.results
cal_fps = lambda x: 1e3/sum(x.speed.values())
fps = [cal_fps(res) for res in results]
print(f'\nMean Inference speed: {sum(fps)/len(fps):.3f} fps\n')

if args.save_path:
    save_dir = args.save_path
    if '.' in os.path.basename(args.save_path): # save_path is a file
        save_dir, file_name = os.path.split(args.source)
        if len(results)>1:
            save_dir = os.path.join(save_dir, os.path.splitext(file_name)[-1])
    os.makedirs(save_dir, exist_ok=True)

    # Process results list
    for res_idx, result in enumerate(results):
        # boxes = result.boxes  # Boxes object for bounding box outputs
        # probs = result.probs  # Probs object for classification outputs
        save_path = os.path.join(save_dir, f'{res_idx:03d}.png')
        result.plot(line_width=2, 
                    font_size=2,
                    save=True,
                    filename=save_path)
        
    print(f'Result saved to {save_dir}')
