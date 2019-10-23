import os
import pandas as pd
import cv2 as cv

from gluoncv import model_zoo
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord

# Initial steps
path_to_videos = "/home/rcuello/SecondDisk/CloudMounting/GD/Exercise_full"
# Loading pre-trained models
detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
pose_net = model_zoo.get_model('simple_pose_resnet18_v1b', pretrained=True)

detector.reset_class(["person"], reuse_weights=['person'])


def get_video_info(source_path):
    file_list = os.listdir(source_path)
    dict_holder = {
            "PoseClipId":0,
            "FileName": "",
            "VideoSourceId":0,
            "ExerciseType" : "",
            "ClipNumber":0,
            "SampleType":""
    }
    df = pd.DataFrame(columns=['PoseClipId','FileName','VideoSourceId','ExerciseType','ClipNumber','SampleType'])
    
    
    for video_file in file_list:
        spl_fname = video_file.split(sep='_')
        dict_holder["PoseClipId"] = abs(hash(video_file)) % (10 ** 8)
        dict_holder["FileName"] = video_file
        dict_holder["VideoSourceId"] = spl_fname[0]
        dict_holder["ExerciseType"] = spl_fname[1]
        dict_holder["ClipNumber"] = spl_fname[2]
        dict_holder["SampleType"] = spl_fname[3].split(sep='.')[0]
        df = df.append(dict_holder,ignore_index=True)
        
    return(df)            

def process_frame(video_frame):
    # x, img = data.transforms.presets.ssd.load_test(video_frame, short=512)
    x, img = data.transforms.presets.ssd.transform_test(video_frame, short=512)
    print('Shape of pre-processed frame:', x.shape)
    class_IDs, scores, bounding_boxs = detector(x)
    pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs)
    predicted_heatmap = pose_net(pose_input)
    pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
    return(pred_coords)
    
    
def process_video_file(source_path,video_file):
    full_path = os.path.join(source_path,video_file)
    video_cap = cv.VideoCapture(full_path)
    n_fps = int(video_cap.get(cv.CAP_PROP_FPS))
    cnt = 1
    while (video_cap.isOpened()):
        ret, frame = video_cap.read()
        if ret == True:
            if cnt==n_fps:
                key_points = process_frame(frame)
                cnt=1
            else:
                cnt+=1
        else:
            break
    # releasing video file
    video_cap.release()
