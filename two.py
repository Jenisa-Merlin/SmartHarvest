import json

def extract_bounding_boxes(annotation):
    bounding_boxes = []

    for obj in annotation["objects"]:
        exterior = obj["points"]["exterior"]
        x_min = min(exterior[0][0], exterior[1][0])
        y_min = min(exterior[0][1], exterior[1][1])
        x_max = max(exterior[0][0], exterior[1][0])
        y_max = max(exterior[0][1], exterior[1][1])
        bounding_boxes.append((x_min, y_min, x_max, y_max))

    return bounding_boxes

# Load annotation from file
file_path = "ann\CropLoadEstimation_image-31.png.json"
with open(file_path, "r") as f:
    annotation = json.load(f)

# Extract bounding boxes
bounding_boxes = extract_bounding_boxes(annotation)
print("Bounding boxes coordinates:", bounding_boxes)
