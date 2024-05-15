import cv2
import argparse
import numpy as np
import json
import os

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = tuple(map(int, COLORS[class_id]))
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def load_classes(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def preprocess_image(image, target_size=(416, 416)):
    return cv2.dnn.blobFromImage(image, 1/255.0, target_size, swapRB=True, crop=False)

def load_model(config_path, weights_path):
    return cv2.dnn.readNet(weights_path, config_path)

def detect_objects_in_image(image, model, classes, conf_threshold=0.5, nms_threshold=0.4):
    Height, Width = image.shape[:2]
    blob = preprocess_image(image)
    model.setInput(blob)
    outs = model.forward(get_output_layers(model))

    detected_objects = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w // 2
                y = center_y - h // 2
                detected_objects.append({"bbox": [x, y, w, h], "class": classes[class_id], "confidence": confidence})

    return detected_objects

def detect_objects(image, net, classes, conf_threshold=0.5, nms_threshold=0.4):
    Height, Width = image.shape[:2]
    blob = preprocess_image(image)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w // 2
                y = center_y - h // 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            draw_prediction(image, class_ids[i], confidences[i], x, y, x + w, y + h)
            
def save_image_with_boxes(image, output_dir, output_file):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    cv2.imwrite(output_path, image)
    print(f"Image with detected objects saved at: {output_path}")

def load_annotations(annotation_dir, image_name):
    annotations = {}
    print(annotation_dir)
    annotation_file = os.path.join(annotation_dir, f"{image_name}.png.json")
    print(annotation_file)
    if os.path.isfile(annotation_file):
        with open(annotation_file) as f:
            data = json.load(f)
            annotations[image_name] = data["objects"]
    else:
        print(f"No annotation file found for image: {image_name}")
    print(annotations)
    return annotations

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

def evaluate(gt_annotations, image_with_boxes):
    total_iou = 0
    total_objects = 0

    for image_name, gt_objects in gt_annotations.items():
        if image_name in image_with_boxes:
            detected_objects = image_with_boxes[image_name]

            for gt_obj in gt_objects:
                gt_box = gt_obj["points"]["exterior"]
                gt_bbox = [gt_box[0][0], gt_box[0][1], gt_box[1][0], gt_box[1][1]]

                for detected_obj in detected_objects:
                    detected_box = detected_obj["bbox"]
                    detected_bbox = [detected_box[0], detected_box[1], detected_box[0] + detected_box[2], detected_box[1] + detected_box[3]]

                    iou = calculate_iou(gt_bbox, detected_bbox)

                    if iou > 0.5:
                        total_iou += iou
                        total_objects += 1
                        break  # Only count each detected object once

    mean_iou = total_iou / total_objects if total_objects > 0 else 0

    return mean_iou
def main(args):
    global classes, COLORS
    classes = load_classes(args.classes)
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

    model = load_model(args.config, args.weights)
    image = cv2.imread(args.image)
    detect_objects(image, model, classes)
    
    # Output directory and file name
    output_dir = 'output'
    output_file = 'detected_objects.jpg'

    # Save image with bounding boxes
    save_image_with_boxes(image, output_dir, output_file)  # Pass the modified image here

    cv2.imshow("Object Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Load ground truth annotations for the input image
    gt_annotations = load_annotations(args.annotation_dir, os.path.splitext(os.path.basename(args.image))[0])

    # Run inference on evaluation images (Assuming you already have this part implemented)
    pred_annotations = {}  # Dictionary mapping image names to predicted objects

    # Evaluate model
    mean_iou = evaluate(gt_annotations, pred_annotations)
    print("Mean IoU:", mean_iou)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help='img')
    ap.add_argument('-c', '--config', required=True, help='yolo_v3.cfg')
    ap.add_argument('-w', '--weights', required=True, help='yolov3.weights')
    ap.add_argument('-cl', '--classes', required=True, help='yolov3.txt')
    ap.add_argument('-ann', '--annotation_dir', required=True, help='ann')
    args = ap.parse_args()
    main(args)