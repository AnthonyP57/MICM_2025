import tensorflow as tf
import cv2
import time
import os
import numpy as np

import posenet
from posenet.constants import *
from posenet.decode_multi import decode_multiple_poses
import posenet.converter.config

import tensorflow.compat.v1 as tf  #dla tensorflow 1...
tf.disable_v2_behavior()
tf.disable_eager_execution()

model=101
# cam_id=1
cam_id = '/home/anthonyp57/VSCode_projects/MICM_2025/lab_3/man_vid.mp4'
cam_width=1280
cam_height=720

MODEL_DIR = '/home/anthonyp57/VSCode_projects/interakcja/lab_3/_models'
DEBUG_OUTPUT = False

def load_model(model_id, sess, model_dir=MODEL_DIR):
    model_ord = 3
   # model_cfg = load_config(model_ord)
    converter_cfg = posenet.converter.config.load_config()
    checkpoints = converter_cfg['checkpoints']
    output_stride = converter_cfg['outputStride']
    checkpoint_name = checkpoints[model_ord]

    model_cfg = {
        'output_stride': output_stride,
        'checkpoint_name': checkpoint_name,
    }
    model_path = os.path.join(model_dir, 'model-%s.pb' % model_cfg['checkpoint_name'])
    if not os.path.exists(model_path):
        print('Cannot find model file %s, converting from tfjs...' % model_path)
        from posenet.converter.tfjs2python import convert
        convert(model_ord, model_dir, check=False)
        assert os.path.exists(model_path)

    with tf.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    if DEBUG_OUTPUT:
        graph_nodes = [n for n in graph_def.node]
        names = []
        for t in graph_nodes:
            names.append(t.name)
            print('Loaded graph node:', t.name)

    offsets = sess.graph.get_tensor_by_name('offset_2:0')
    displacement_fwd = sess.graph.get_tensor_by_name('displacement_fwd_2:0')
    displacement_bwd = sess.graph.get_tensor_by_name('displacement_bwd_2:0')
    heatmaps = sess.graph.get_tensor_by_name('heatmap:0')

    return model_cfg, [heatmaps, offsets, displacement_fwd, displacement_bwd]

def get_adjacent_keypoints(keypoint_scores, keypoint_coords, min_confidence=0.1):
    results = []
    for left, right in posenet.constants.CONNECTED_PART_INDICES:
        if keypoint_scores[left] < min_confidence or keypoint_scores[right] < min_confidence:
            continue
        results.append(
            np.array([keypoint_coords[left][::-1], keypoint_coords[right][::-1]]).astype(np.int32),
        )
    return results

def draw(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_score=0.5, min_part_score=0.5, angle=None, pos=None, hand=None):
    out_img = img
    adjacent_keypoints = []
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_score:
            continue

        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score)
        adjacent_keypoints.extend(new_keypoints)

        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_score:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))

    out_img = cv2.drawKeypoints(
        out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    
    if all([angle, pos, hand]):
        cv2.putText(out_img, f'shoulder-elbow-wrist angle: {angle:.2f}', (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out_img, f'hand: {hand}', (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out_img, f'acceleration: {pos:.2f}', (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
        
    else:
        cv2.putText(out_img, 'arm not straight', (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
        
    return out_img

def shoulder_elbow_wrist_angle(shoulder, elbow, wrist):
    shoulder = np.array(shoulder)
    elbow = np.array(elbow)
    wrist = np.array(wrist)

    shoulder_elbow = elbow - shoulder
    shoulder_elbow /= np.linalg.norm(shoulder_elbow) #normalize by norm to = 1

    shoulder_wrist = wrist - shoulder
    shoulder_wrist /= np.linalg.norm(shoulder_wrist)

    angle = np.arccos(np.clip(np.dot(shoulder_elbow, shoulder_wrist), -1.0, 1.0)) #cos similarity inverse so to speak, clip for stability
    return np.degrees(angle) #angle in degrees

def main():
    old_y_pos = None
    with tf.Session() as sess:
        model_cfg, model_outputs = load_model(model, sess)
        output_stride = model_cfg['output_stride']

        cap = cv2.VideoCapture(cam_id)

        cap.set(3, cam_width)
        cap.set(4, cam_height)

        start = time.time()
        frame_count = 0
        while True:
            if frame_count % 10 == 0:

                res, img = cap.read()
                if not res:
                    raise IOError("webcam failure")
                        
                width=img.shape[1]
                height=img.shape[0]
                target_width = (int(width) // output_stride) * output_stride + 1
                target_height = (int(height) // output_stride) * output_stride + 1
                scale = np.array([img.shape[0] / target_height, img.shape[1] / target_width])

                input_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
                input_img = input_img * (2.0 / 255.0) - 1.0
                input_img = input_img.reshape(1, target_height, target_width, 3)
                display_image=img.copy()

                heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                    model_outputs,
                    feed_dict={'image:0': input_img}
                )

                pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                    heatmaps_result.squeeze(axis=0),
                    offsets_result.squeeze(axis=0),
                    displacement_fwd_result.squeeze(axis=0),
                    displacement_bwd_result.squeeze(axis=0),
                    output_stride=output_stride,
                    max_pose_detections=10,
                    min_pose_score=0.15)

                keypoint_coords *= scale
                nose = keypoint_coords[0, PART_NAMES.index('nose')]
                left_shoulder = keypoint_coords[0, PART_NAMES.index('leftShoulder')]
                right_shoulder = keypoint_coords[0, PART_NAMES.index('rightShoulder')]
                left_elbow = keypoint_coords[0, PART_NAMES.index('leftElbow')]
                right_elbow = keypoint_coords[0, PART_NAMES.index('rightElbow')]
                left_wrist = keypoint_coords[0, PART_NAMES.index('leftWrist')]
                right_wrist = keypoint_coords[0, PART_NAMES.index('rightWrist')]

                left_angle = shoulder_elbow_wrist_angle(left_shoulder, left_elbow, left_wrist)
                right_angle = shoulder_elbow_wrist_angle(right_shoulder, right_elbow, right_wrist)
                
                if left_angle < 20 or right_angle < 20:
                    side = 'left' if left_angle < right_angle else 'right'
                    if side == 'left':
                        wrist_abose_nose_y_pos = left_wrist[1] - nose[1]
                        angle = left_angle
                    else:
                        wrist_abose_nose_y_pos = right_wrist[1] - nose[1]
                        angle = right_angle
                else:
                    wrist_abose_nose_y_pos = None
                    angle = None
                    side = None

                if old_y_pos is None:
                    old_y_pos = wrist_abose_nose_y_pos

                if wrist_abose_nose_y_pos is not None:
                    wrist_abose_nose_y_pos -= old_y_pos
                
                overlay_image =draw(
                    display_image, pose_scores, keypoint_scores, keypoint_coords,
                    min_pose_score=0.15, min_part_score=0.1,
                    angle=angle, pos=wrist_abose_nose_y_pos, hand=side)
                
                cv2.imshow('posenet', overlay_image)
                frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            print('Average FPS: ', frame_count / (time.time() - start), end='\r')
  

if __name__ == "__main__":
    main()
