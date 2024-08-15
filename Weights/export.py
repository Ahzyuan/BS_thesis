import argparse,os
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('-p','--pt_model',type=str, default='/home/yzq/hzy/yolov8/runs/detect/train/weights/best.pt', help='pytorch model path')
parser.add_argument('-s','--save_path',type=str, default='/home/yzq/hzy/yolov8/export_models/A1', help='onnx model save path')
parser.add_argument('-f','--format',type=str, default='o', help='export model format, o for onnx and t for engine')
parser.add_argument('-r','--hw_ratio',type=float, default=0.6, help='common value of height / width of img in real scenario')
parser.add_argument('--workspace',type=int, default=4096, help='workspace for trtexec')
parser.add_argument('--half', action='store_true', default=False, help='half precision')
parser.add_argument('--dynamic',action='store_true', default=False, help='dynamic input size')
parser.add_argument('--simplify', action='store_true', default=False, help='simplify onnx model')
parser.add_argument('--verbose',action='store_true', default=False, help='enable verbose mode in trtexec')
args = parser.parse_args()
assert os.path.exists(args.pt_model), f'pt model of which path is {args.pt_model} not exist'
assert args.format in ['o','t'], 'format must be o or t, o for onnx while t for engine'
assert not (args.half and args.dynamic), 'Half precision and dynamic input size can not be set at the same time'

# Load a model
model = YOLO(args.pt_model) 

# Get export file name
if '.' not in os.path.basename(args.save_path): # save_path is a dir
    os.makedirs(args.save_path,exist_ok=True)
    file_name = ['yolo8', '.onnx' if args.format == 'o' else '.engine']
else:
    args.save_path, file_name = os.path.split(args.save_path)
    file_name = list(os.path.splitext(file_name))
    if file_name[-1] == '.onnx':
        args.format = 'o'
    elif file_name[-1] == '.engine':
        args.format = 't'
    else:
        raise ValueError('save path must end with .onnx or .engine')

## Get file suffix
suffix = []
for argi in ('half', 'dynamic', 'simplify'):
    if getattr(args, argi):
        suffix.append('_' + (argi if argi!='simplify' else 'sim'))

## Set imgsz according to common used height / width ratio in real scenario
align2stride = lambda x,s: (x//s)*s
max_spatial_size = model.model.args['imgsz'] # 640 as usual
max_sride = int(max(model.model.stride)) # 32 as usual
if args.hw_ratio < 1:
    w = int(max_spatial_size)
    h = int(align2stride(w * args.hw_ratio, max_sride))
else:
    h = int(max_spatial_size)
    w = int(align2stride(h * args.hw_ratio, max_sride))
suffix.append(f'_{h}x{w}')

file_name.insert(-1, ''.join(suffix))
file_name = ''.join(file_name)
dst_path = os.path.join(args.save_path, file_name)

# Export the model
export_file_path = model.export(format="onnx" if args.format == 'o' else 'engine',
                                imgsz = [h,w], # int(max_spatial_size), # int, [h,w] / str, '[h,w]'
                                dynamic=args.dynamic,
                                half=args.half,
                                simplify=args.simplify,
                                workspace = args.workspace, 
                                verbose = args.verbose)

os.system(f'mv {export_file_path} {dst_path}')
if args.format == 't':
    os.system(f'rm {args.pt_model[:-3]}.onnx')

print(f'Exported model saved to {dst_path}')