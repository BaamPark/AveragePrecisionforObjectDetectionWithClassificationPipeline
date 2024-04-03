import os
from pathlib import Path
from typing import List, Dict, Optional
from lxml import etree
import re
from ultralytics import YOLO
from PIL import Image
import cv2
from transformers import pipeline
from logger_config import logger
import matplotlib.pyplot as plt 
import pandas as pd


def main():
    videoset_path = Path("/home/beomseok/ppe_data/dataset_test/ppe vanishing pen simulation-trauma 1 above bed")
    # videoset_path = Path("/home/beomseok/ppe_data/dataset_test/wanzhao sim-trauma 1 above bed")
    # videoset_path = Path("/home/beomseok/ppe_data/dataset_test/2.24.23 sim without bed-trauma 1 above bed")
    h_ga_gi_detector = YOLO("/home/beomseok/PPE_cleaned/ultralytics/runs/detect/data_collorjitter_aug_exp_with_11sets/weights/best.pt")
    mask_classifier = pipeline("image-classification", model="stage2/transformer_results/dir25", device=0)
    obj_cls_list = ["rc", "nc", "ma", "mi"]

    frames_folder_path = videoset_path / 'Frames'
    labels_folder_path = videoset_path / 'Labels'

    sorted_frames_folder_path = sorted(frames_folder_path.iterdir(), key=extract_number)

    img_lbl_dict = {"img":[], "lbl":[]}

    for file in sorted_frames_folder_path:
        if file.suffix in ['.jpg', '.png']:
            img_lbl_dict['img'].append(file)
            img_lbl_dict['lbl'].append(labels_folder_path / (file.stem + '.xml'))

    plt.figure(figsize=(10, 7))  # Create a new figure
    ax = plt.gca()  # Get the current Axes instance

    ap_for_each_cls_dict = {}

    for obj_cls in obj_cls_list:
        predictions_df = pd.DataFrame(columns=['confidence', 'is_tp', 'is_fp'])
        num_total_gt_boxes = 0
        logger.info(f"current object class: {obj_cls}")
        for i in range(len(img_lbl_dict['lbl'])):
            gt_bboxes_per_frame = extract_bbox_from_xml(img_lbl_dict['lbl'][i], img_lbl_dict['img'][i], obj_cls)
            pred_bboxes_with_conf_per_frame = extract_predictition_on_frame(img_lbl_dict['img'][i], obj_cls, h_ga_gi_detector, mask_classifier)
            num_total_gt_boxes += len(gt_bboxes_per_frame)
            logger.info(f"xml file name: {img_lbl_dict['lbl'][i]}")
            logger.info(f"gt_bboxes_per_frame, {gt_bboxes_per_frame}")
            logger.info(f'pred_bboxes_per_frame, {pred_bboxes_with_conf_per_frame}')
            predictions_df = compute_confusion_matrix_by_iou(gt_bboxes_per_frame, pred_bboxes_with_conf_per_frame, predictions_df)

        sorted_predictions_df = predictions_df.sort_values(by='confidence', ascending=False)
        ap_score, precision_list, recall_list = compute_ap11_score(sorted_predictions_df, num_total_gt_boxes)
        ap_for_each_cls_dict[obj_cls] = ap_score

        plot_precision_recall_curve(precision_list, recall_list, ap_score, obj_cls, ax)
        
        print({k: round(v, 3) for k, v in ap_for_each_cls_dict.items()})

    ax.legend(loc='lower left')  # Add a legend to the plot
    plt.savefig('pr_curve/precision_recall_curve.png')  # Save the figure
    plt.close()  # Close the figure window
    return



def extract_bbox_from_xml(xml_path: Path, img_path: Path, obj_cls: str) -> List[Dict[str, int]]:
    # Parse the XML file using lxml
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    bboxes = []

    image = cv2.imread(str(img_path))
    height, width, channels = image.shape

    # Iterate over all 'object' elements in the XML
    for obj in root.findall('object'):
        name = obj.find('name').text

        # Check if the object's name matches the specified class
        if obj_cls in name and name[0] == "h":
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > width:
                xmax = width
            if ymax > height:
                ymax = height

            # Append the bounding box coordinates to the list
            bboxes.append({'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
    
    return bboxes


def extract_predictition_on_frame(img_path: Path, obj_cls: str, detector: YOLO, classifier) -> List[Dict[str, int]]:
    box_conf_dict_list = []
    result = detector(img_path)[0]

    input_img  = result.orig_img

    for box in result.boxes:
        if box.cls == 0: #when pred bbox is head
            xmin, ymin, xmax, ymax = [int(x) for x in box.xyxy[0].tolist()]
            detection_conf = box.conf.item() #YOLO returns bbox in descending order of confidence
            cropped_frame = input_img[ymin:ymax, xmin:xmax]
            cropped_frame = Image.fromarray(cropped_frame[..., ::-1])

            conf_and_pred = classifier(cropped_frame)[0]
            classification_pred, classification_conf = conf_and_pred['label'], conf_and_pred['score']

            if classification_pred == obj_cls:
                box_conf_dict_list.append({"box": {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}, "conf": detection_conf * classification_conf})

    return box_conf_dict_list


def compute_confusion_matrix_by_iou(gt_bboxes_per_frame: list, pred_bboxes_with_conf_per_frame:list, predictions_df: pd.DataFrame, iou_threshold=0.5):

    def calculate_iou(bbox1, bbox2):
    # Extract the coordinates of the bounding boxes
        xmin1, ymin1, xmax1, ymax1 = bbox1['xmin'], bbox1['ymin'], bbox1['xmax'], bbox1['ymax']
        xmin2, ymin2, xmax2, ymax2 = bbox2['xmin'], bbox2['ymin'], bbox2['xmax'], bbox2['ymax']
        
        # Calculate the coordinates of the intersection rectangle
        ixmin = max(xmin1, xmin2)
        iymin = max(ymin1, ymin2)
        ixmax = min(xmax1, xmax2)
        iymax = min(ymax1, ymax2)
        
        # Calculate the area of the intersection rectangle
        iw = max(0, ixmax - ixmin + 1)
        ih = max(0, iymax - iymin + 1)
        intersection_area = iw * ih
        
        # Calculate the area of each bounding box
        area1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
        area2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)
        
        # Calculate the area of the union
        union_area = area1 + area2 - intersection_area
        
        # Avoid division by zero
        if union_area == 0:
            return 0
        
        # Calculate IoU
        iou = intersection_area / union_area
        return iou
    

    #this dictionary stores index of each bounding box set and the matched iou
    matched_box_with_iou = {
        "gt": [],
        "pred": [],
        "iou":[]
    }

    for i, pd_conf_box in enumerate(pred_bboxes_with_conf_per_frame):

        iou_list = []
        for gt_box in gt_bboxes_per_frame:
            iou = calculate_iou(pd_conf_box['box'], gt_box)
            iou_list.append(iou)

        if not iou_list:
            continue

        max_iou, max_iou_idx = max((value, index) for index, value in enumerate(iou_list))
        
        if max_iou > iou_threshold:
            if not max_iou_idx in matched_box_with_iou['gt']:
                matched_box_with_iou["gt"].append(max_iou_idx)
                matched_box_with_iou['pred'].append(i)
                matched_box_with_iou['iou'].append(max_iou)
            else: #when ground truth already has match with one of previous bbox
                gt_index = matched_box_with_iou['gt'].index(max_iou_idx)
                #if the matched ground truth has smaller iou than the current iou
                if max_iou > matched_box_with_iou['iou'][gt_index]:
                    # logger.info(f"the gt box: {gt_bboxes_per_frame[gt_index]} already has match with pred box: {pred_bboxes_per_frame[gt_index]}")
                    # logger.info(f"But current IoU is bigger than exsiting one, so updating the index of predbox")
                    matched_box_with_iou['pred'][gt_index] = i
                    matched_box_with_iou['iou'][gt_index] = max_iou

    for pred_box_index in matched_box_with_iou['pred']:

        predictions_df = predictions_df.append({
            'confidence': pred_bboxes_with_conf_per_frame[pred_box_index]['conf'],
            'is_tp': True,
            'is_fp': False
        }, ignore_index=True)

    tp_predbox_index_set = set(matched_box_with_iou['pred'])
    fp_predbox_conf_list = [pred_bboxes_with_conf_per_frame[index]['conf'] for index in range(len(pred_bboxes_with_conf_per_frame)) if index not in tp_predbox_index_set]

    for fp_predbox_conf in fp_predbox_conf_list:

        predictions_df = predictions_df.append({
            'confidence': fp_predbox_conf,
            'is_tp': False,
            'is_fp': True
        }, ignore_index=True)

    return predictions_df

# AP score with 11 points interpolation
def compute_ap11_score(sorted_predictions_df:pd.DataFrame, num_total_gt_boxes:int):
    num_total_tp = sorted_predictions_df['is_tp'].sum() 
    total_fn = num_total_gt_boxes - num_total_tp

    precision_list = []
    recall_list = []
    cumulative_tp = 0
    cumulative_fp = 0
    ap11_list = []
    ap11_precision_list = []
    ap11_recall_list = [i/10 for i in range(11)]

    for index, row in sorted_predictions_df.iterrows():
        if row['is_tp']:
            cumulative_tp += 1
        else:
            cumulative_fp += 1

        if cumulative_tp + cumulative_fp == 0:
            precision = 0
        else:
            precision = cumulative_tp / (cumulative_tp + cumulative_fp)

        if cumulative_tp + total_fn == 0:
            recall = 0
        else:
            recall = cumulative_tp / (cumulative_tp + total_fn)

        precision_list.append(precision)
        recall_list.append(recall)

    for recall_threshold in ap11_recall_list:
        relevant_precisions = [p for r, p in zip(recall_list, precision_list) if r >= recall_threshold]
        if relevant_precisions:
            ap11_precision_list.append(max(relevant_precisions))
        else:
            ap11_precision_list.append(0.0)

    ap11_score = sum(ap11_precision_list) / 11
    return ap11_score, ap11_precision_list, ap11_recall_list

def compute_ap_score(sorted_predictions_df:pd.DataFrame, num_total_gt_boxes:int):
    num_total_tp = sorted_predictions_df['is_tp'].sum() 
    total_fn = num_total_gt_boxes - num_total_tp

    precision_list = []
    recall_list = []
    cumulative_tp = 0
    cumulative_fp = 0
    prev_recall = 0
    ap_score = 0

    for index, row in sorted_predictions_df.iterrows():
        if row['is_tp']:
            cumulative_tp += 1
        else:
            cumulative_fp += 1

        if cumulative_tp + cumulative_fp == 0:
            precision = 0
        else:
            precision = cumulative_tp / (cumulative_tp + cumulative_fp)

        if cumulative_tp + total_fn == 0:
            recall = 0
        else:
            recall = cumulative_tp / (cumulative_tp + total_fn)
        
        ap_score += (recall - prev_recall) * precision
        prev_recall = recall

        if ap_score > 1:
            logger.warning(f"invalid ap score: {ap_score}")

        precision_list.append(precision)
        recall_list.append(recall)
    

    return ap_score, precision_list, recall_list


def plot_precision_recall_curve(precision_list, recall_list, ap_score, obj_cls, ax):
    ax.plot(recall_list, precision_list, marker='.', label=f'{obj_cls} (AP: {ap_score:.3f})')
    ax.set_title('Precision-Recall Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.grid(True)

    # plt.figure(figsize=(10, 6))
    # plt.plot(recall_list, precision_list, marker='.')
    # plt.title(f'Precision-Recall curve for {obj_cls}')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.grid(True)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.legend([f'{obj_cls} AP: {ap_score:.2f}'])

    # filename = f'pr_curve_{obj_cls}.png'
    # filepath = os.path.join('pr_curve', filename)
    # plt.savefig(filepath)
    # plt.close()  # Close the figure window


def extract_number(file_path: Path) -> int:
    # Extract the trailing number from the file name
    match = re.search(r'(\d+)\.\w+$', file_path.name)
    return int(match.group(1)) if match else 0


if __name__ == "__main__":
    main()