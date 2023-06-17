# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import smtplib
import sys
import threading
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms import Compose
from utils.general import TryExcept

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
ROOTP = FILE.parents[1]  # Parent directory of yolov5, which contains the midas module
midas_directory = os.path.join(ROOTP,  "MiDaS")
if str(ROOTP) not in sys.path:
    sys.path.append(str(midas_directory))  # Add ROOT to PATH


import model_loader
import midas
from midas import transforms
from midas.transforms import NormalizeImage, PrepareForNet, Resize

from models.common import DetectMultiBackend
from scipy.optimize import least_squares
from utils.dataloaders import (IMG_FORMATS, VID_FORMATS, LoadImages,
                               LoadScreenshots, LoadStreams)
from utils.general import (LOGGER, Profile, check_file, check_img_size,
                           check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args,
                           scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
depth_image=None
counter = 0
Stop_run=False
#xy=[]
S=None
Depth = 0
depth_map= None
scale,shift= 0,0
detected=False

def sendEmail():
    # Email configuration
    sender_email = 'FireDetectionRT@outlook.com'
    receiver_email = email_addresses[0]#'91830054@students.liu.edu.lb'
    password = 'Firedetection'
    smtp_server = 'smtp-mail.outlook.com'
    smtp_port = 587

    # Create a multipart message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = 'Fire has been detected'
    import datetime

    # Get current date
    current_date = datetime.date.today()
    subject = 'Fire has been detected'
    message = 'Date: '+str(current_date)+'\nFire has been detected, please contact the authorities, raise the alarm and evacuate the building.'
    # Attach the message to the MIMEMultipart object
    msg.attach(MIMEText(message, 'plain'))

    try:
        # Create a secure connection to the SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        # Login to the sender's email account
        server.login(sender_email, password)
        # Send the email
        server.sendmail(sender_email, receiver_email, msg.as_string())
        # Print success message
        print('Email sent successfully!')
    except Exception as e:
        # Print error message
        print(f'Error: {str(e)}')
    finally:
        # Close the SMTP connection
        server.quit()
    return

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=True,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        midas_model= 0, #choose midas model
        alpha=1, #
        timer=0,
        
):
    device= select_device(device)
    global transform, midasM
    if midas_model == 0:
        model_type = "midas_v21_small_256"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
        model_path = str(ROOTP)+'\\midas_v21_small_256.pt'
    else:
        model_type="dpt_beit_large_512" # MiDaS v3.1 (highest accuracy, slowest inference speed) 
        model_path = str(ROOTP)+'\\dpt_beit_large_512.pt' 
    from model_loader import default_models, load_model 
    midasM, transform, net_w, net_h = load_model(device, model_path, model_type, False, False, False)
    midasM.to(device)
    midasM.eval()
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid" or model_type== "dpt_beit_large_512":
        transform = Compose(
            [
                lambda img: {"image": img / 255.0},
                Resize(
                    512,
                    512,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                PrepareForNet(),
                lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
            ]
        )
    else:
        transform = Compose(
            [
                lambda img: {"image": img / 255.0},
                Resize(
                    256,
                    256,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="upper_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
                lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
            ]
        )
   
    # File path to read points from
    points_file = "Refrence_points.txt"
    # Initialize lists to store reference points and email addresses
    global points,email_addresses
    points = []
    email_addresses = []

    # Read data from file
    with open(points_file, 'r') as f:
        lines = f.readlines()

    # Process each line and extract the data
    is_reference_points = False
    is_email_address = False

    for line in lines:
        # Remove leading/trailing whitespace
        line = line.strip()

        # Check if the line contains reference points or email addresses
        if line == "Reference Points:":
            is_reference_points = True
            is_email_address = False
        elif line == "Email Address:":
            is_reference_points = False
            is_email_address = True
        elif is_reference_points:
            components = line.split(",")

            # Check if the line contains all the required components
            if len(components) == 3:
                try:
                    # Convert components to integers/floats and create the reference point
                    point = [int(components[0]), int(components[1]), float(components[2])]

                    # Add the reference point to the list
                    points.append(point)
                except ValueError:
                    # Skip the line if there is an error converting to int/float
                    continue
        elif is_email_address:
            email_addresses.append(line)

    # Convert reference points list to NumPy array
    points = np.array(points)

    # Print the reference points
    print("Reference Points:")
    print(points)

    # Print the email addresses
    print("Email Addresses:")
    for email in email_addresses:
        print(email)
 
    
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download
  
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device =select_device(device)
    model = DetectMultiBackend(weights,device=device ,dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs
    prev_frame_time = 0
    new_frame_time = 0
    global old_xyxy,centery,centerx
    old_xyxy=[0,0,0,0]
    centerx,centery=0,0

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
        
        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            #cv2.imwrite("testing2.png",im0)
            if timer == 0:
                global depth_image
                depth_image=im0.copy()
                run_midas(depth_image)
                timer=1
            global detected,Depth
            detected = False
            Depth=0
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("pressed the key")
                    alpha=0
                    break
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            new_frame_time = time.time()
            fps_vid = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time

            
             
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                
                depth_image=im0.copy()
                
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    detected=True
                    #global xy
                    #xy=xyxy
                    if midas_model != 0:
                        t1,t2,t3,t4 = xyxy
                        t1old,t2old,t3old,t4old = old_xyxy
                        tmid=(t1+t3)/2
                        tmidold=(t1old+t3old)/2
                        if tmid >(tmidold+90) or tmid<(tmidold-90):
                            print(xyxy[0])
                            old_xyxy=xyxy
                            run_midas(depth_image)
                    else:
                        if timer == 1 :
                            timer =60
                            sendEmail()
                            depth = timer60() 
                
                    def depth_to_distance(depth) -> float:
                        return  scale* depth + shift
                    x1, y1, x2, y2 = xyxy

                    center_x = (x1.item() + x2.item()) / 2
                    center_y = (y1.item() + y2.item()) / 2
                    center_y = (y2.item() + center_y) / 2                  
                    centerx=center_x
                    centery=center_y

                    new_point_distance = depth_map[int(center_y), int(center_x)]
                    Depth = depth_to_distance(new_point_distance) 
                    
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                fps_vid = int(fps_vid)
                fps_vid = str(fps_vid)
                im1=im0.copy()
                cv2.putText(im0, fps_vid, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 2, cv2.LINE_AA)
                Depth1="Distance to Fire: " + str(round(Depth,2))+"m" #+str(((Depth-4.10)/4.10)*100)
                cv2.putText(im0, Depth1, (7, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 150), 2, cv2.LINE_AA)
                #if centerx!=0:
                    #cv2.putText(im0, Depth1, (int(centerx), int(centery)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 150), 2, cv2.LINE_AA)
                
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
                
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = 20 #vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 20, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
            
        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        if alpha == 0:
                S.cancel()
                break
        


    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    global Stop_run
    Stop_run=True   
    S.cancel()
# Define run_midas()
def run_midas(img):
    # MiDaS code
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        start_time = time.time()
        prediction = midasM(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth = prediction.cpu().numpy()
    end_time = time.time()
    inference_time = end_time - start_time
    #print("Inference Time:", inference_time, "seconds")
    global depth_map #remove if not needed 
    depth_map = depth
    global counter
    if counter== 0:
        newSNS= SNSCalc(depth)
        global scale,shift
        scale, shift = newSNS.x
        #print(scale)
        #print(shift)
        counter=counter+1
        return


    return depth   

def SNSCalc(depth,):
     # Define the function that calculates the residuals
    def residuals(params, points):
        scale, shift = params
        residuals = []
        for x, y, d in points:
            x, y = map(int, [x, y])
            r = calc_distance(depth_map[y, x], scale, shift) - d
            residuals.append(r)
        return np.array(residuals)

    def calc_distance(depth, scale, shift):
        return depth * scale + shift
    

    # Set up depth map and points
    depth_map = depth 

    if points.shape[1] != 3:
        raise ValueError('Points should have shape (N, 3)')

    # Set up initial parameter values
    init_params = np.array([1, 0])

    # Run least squares optimization to find scale and shift
    result = least_squares(residuals, init_params, args=(points,))
    scale, shift = result.x
    
    return result

def timer60():
        # call this function again in 1.0 seconds
        global S
        S=threading.Timer(1.0, timer60)
        if Stop_run == True:
            S.cancel()
            return
        
        if detected== True:
            global counter
            counter = counter + 1
            run_midas(depth_image)
            #print("detected")
            print(counter) 
        S.start()
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--midas-model', type=int, default=0, help='midas model selector')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    #check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

