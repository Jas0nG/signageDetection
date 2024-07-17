import cv2
import numpy as np
from paddleocr import PaddleOCR
import time
import os
import re


class RectCoordinates:
    """
    Represents the bounding box coordinates for detected text.
    """
    def __init__(self, vec):
        self.top_left = vec[0]
        self.top_right = vec[1]
        self.bottom_right = vec[2]
        self.bottom_left = vec[3]

    def cv_top_left(self):
        return int(self.top_left[0]), int(self.top_left[1])

    def cv_bottom_right(self):
        return int(self.bottom_right[0]), int(self.bottom_right[1])

    def bounding_box(self):
        """
        Calculates the bounding box coordinates and ensures they are integers.
        """
        min_x = min(self.top_left[0], self.top_right[0], self.bottom_right[0], self.bottom_left[0])
        min_y = min(self.top_left[1], self.top_right[1], self.bottom_right[1], self.bottom_left[1])
        max_x = max(self.top_left[0], self.top_right[0], self.bottom_right[0], self.bottom_left[0])
        max_y = max(self.top_left[1], self.top_right[1], self.bottom_right[1], self.bottom_left[1])
        return (int(min_x), int(min_y)), (int(max_x), int(max_y))


class TextTarget:
    """
    Represents the target text information including text, confidence, similarity, and coordinates.
    """
    def __init__(self, text, confidence, similarity):
        self.text = text
        self.confidence = confidence
        self.similarity = similarity
        self.coordinates = RectCoordinates([(0, 0), (0, 0), (0, 0), (0, 0)])

    def set_coordinates(self, coordinates):
        self.coordinates = coordinates


def preprocess_image(image_path):
    """
    Preprocesses the image by loading it and converting it to grayscale.
    Args:
        image_path (str): The path to the image file.
    Returns:
        numpy.ndarray: The preprocessed grayscale image.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (640, 480))
    return gray


def run_ocr(image, ocr, debug=False):
    """
    Runs OCR on the given image and filters the result to keep only the Chinese text with high confidence.
    Args:
        image (numpy.ndarray): The preprocessed grayscale image.
        ocr (PaddleOCR): The PaddleOCR instance.
        debug (bool): If True, prints debug information.
    Returns:
        list: The filtered OCR result, containing the coordinates, text, and confidence of each detected line.
    """
    result = ocr.ocr(image, cls=True)
    filtered_result = []
    for res in result:
        for line in res:
            coordinates, (text, confidence) = line
            if re.match(r'[\u4e00-\u9fa5]', text) and confidence > 0.6:
                if debug:
                    print(f"Detected Text: {text} with Confidence: {confidence}")
                filtered_result.append(line)
    return filtered_result


def levenshtein_distance(s1, s2):
    """
    Computes the Levenshtein distance between two strings.
    Args:
        s1 (str): First string.
        s2 (str): Second string.
    Returns:
        int: The Levenshtein distance between the two strings.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_similarity(s1, s2, debug=False):
    """
    Computes the similarity between two strings using Levenshtein distance.
    Args:
        s1 (str): First string.
        s2 (str): Second string.
        debug (bool): If True, prints debug information.
    Returns:
        float: The similarity score between 0 and 1.
    """
    dist = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    similarity = 1 - dist / max_len
    if debug:
        print(f"Compare text: {s1} and {s2}, Score: {similarity}")
    return similarity


def detect_arrow(image, text_target, debug=False):
    """
    Detects the indicator arrow near the given text target.
    Args:
        image (numpy.ndarray): The preprocessed grayscale image.
        text_target (TextTarget): The target text object.
        debug (bool): If True, shows debug information.
    Returns:
        bool: True if the indicator arrow is detected, False otherwise.
    """
    top_left, bottom_right = text_target.coordinates.bounding_box()
    extend_pixel = (bottom_right[0] - top_left[0]) * 2
    top_left = (max(0, top_left[0] - extend_pixel), max(0, top_left[1] - extend_pixel))
    bottom_right = (min(image.shape[1], bottom_right[0] + extend_pixel), min(image.shape[0], bottom_right[1] + extend_pixel))
    roi = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    if debug:
        cv2.imshow("ROI", roi)
        cv2.waitKey(0)

    image_pyramid = [roi]
    while image_pyramid[-1].shape[0] > 100 and image_pyramid[-1].shape[1] > 100:
        image_pyramid.append(cv2.pyrDown(image_pyramid[-1]))

    for roi in image_pyramid:
        roi = cv2.GaussianBlur(roi, (5, 5), 0)
        edges = cv2.Canny(roi, 50, 150, apertureSize=3)
        if debug:
            cv2.imshow("Edges", edges)
            cv2.waitKey(0)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 20)
        # Add logic to process lines

    return False


if __name__ == '__main__':
    image_folder = './image'
    destination_text = "药房"
    debug = True

    ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False, use_gpu=True)
    print("OCR module loaded successfully!")

    for image_name in os.listdir(image_folder):
        if re.match(r'.*\.(jpg|png)', image_name):
            print(f"Processing image: {image_name}")
            image_path = os.path.join(image_folder, image_name)
            start_time = time.time()
            image = preprocess_image(image_path)
            result = run_ocr(image, ocr, debug)
            execution_time = time.time() - start_time

            found_destination_text = False
            text_target = TextTarget(destination_text, 0.0, 0.0)
            mask_image = image.copy()

            for line in result:
                coordinates, (text, confidence) = line
                top_left, bottom_right = RectCoordinates(coordinates).bounding_box()
                cv2.rectangle(mask_image, top_left, bottom_right, (255, 255, 255), -1)
                similarity = compute_similarity(text, destination_text, debug)
                if similarity > 0.0 and not found_destination_text:
                    found_destination_text = True
                    text_target.confidence = confidence
                    text_target.similarity = similarity
                    text_target.set_coordinates(RectCoordinates(coordinates))

            if found_destination_text:
                if debug:
                    image_show = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    print(f"Found target text '{destination_text}' with confidence {text_target.confidence} and similarity {text_target.similarity}")
                    top_left, bottom_right = text_target.coordinates.bounding_box()
                    cv2.rectangle(image_show, text_target.coordinates.cv_top_left(), text_target.coordinates.cv_bottom_right(), (0, 255, 0), 2)
                    cv2.imshow("Result", image_show)
                    cv2.waitKey(0)
                detect_arrow(mask_image, text_target, debug)
            else:
                print(f"Cannot find the target text '{destination_text}'")

            print(f"OCR Execution time: {round(execution_time * 1e3, 2)} ms")
