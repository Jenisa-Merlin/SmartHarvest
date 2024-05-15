def calculate_iou(box1, box2):
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate area of intersection rectangle
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate area of each bounding box
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

def evaluate_iou(gt_boxes, detected_boxes):
    total_iou = 0
    total_pairs = 0

    for gt_box in gt_boxes:
        for detected_box in detected_boxes:
            iou = calculate_iou(gt_box, detected_box)
            total_iou += iou
            total_pairs += 1

    mean_iou = total_iou / total_pairs if total_pairs > 0 else 0

    return mean_iou

# Example usage
detected_boxes = [(971, 198, 1044, 271), (708, 1472, 824, 1588), (116, 1310, 238, 1432), (659, 1865, 694, 1900)]
gt_boxes = [(931, 760, 1097, 931), (393, 713, 562, 871), (237, 372, 374, 533), (692, 117, 835, 272), (135, 1013, 302, 1212), (193, 278, 325, 419), (48, 248, 193, 402), (747, 655, 881, 828), (962, 978, 1116, 1148), (265, 1096, 403, 1270), (257, 1435, 415, 1630)]

mean_iou = evaluate_iou(gt_boxes, detected_boxes)
print("Mean IoU:", mean_iou)
