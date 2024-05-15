import cv2

def extract_boxes_from_image(image_path, output_path):
    # Load the Haar Cascade classifier for detecting faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the image
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Create a copy of the image to draw boxes on
    image_with_boxes = image.copy()

    # List to store the coordinates of bounding boxes
    bounding_boxes = []

    # Draw the bounding boxes on the image and extract their coordinates
    for (x, y, w, h) in faces:
        # Draw the bounding box on the image
        cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Add the bounding box coordinates to the list
        bounding_boxes.append((x, y, x+w, y+h))

    return bounding_boxes

# Example usage
image_path = "img\CropLoadEstimation_image-31.png"
output_path = "data\CropLoadEstimation_image-31.png"
bounding_boxes = extract_boxes_from_image(image_path, output_path)
print("Bounding boxes coordinates:", bounding_boxes)
