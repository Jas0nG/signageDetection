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
        # blur
        roi = cv2.GaussianBlur(roi, (5, 5), 0)
        # OTSU 二值化
        _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 模板匹配
        template = cv2.imread("./template/arrow_R.jpg", cv2.IMREAD_GRAYSCALE)
        # 模板预处理，二值化并留下黑色部分
        _, template = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        template = cv2.bitwise_not(template)
        
        if debug:
            cv2.imshow("Template", template)
            cv2.waitKey(0)
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.5
        loc = np.where(res >= threshold)
        if debug:
            roiShow = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        for pt in zip(*loc[::-1]):
            #gray to bgr
            cv2.rectangle(roiShow, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
            print("Found arrow")
            # return True
        if debug:
            cv2.imshow("Arrow", roiShow)
            cv2.waitKey(0)

        
        # roi = cv2.GaussianBlur(roi, (5, 5), 0)
        # edges = cv2.Canny(roi, 50, 150, apertureSize=3)
        # if debug:
        #     cv2.imshow("Edges", edges)
        #     cv2.waitKey(0)
        # lines = cv2.HoughLines(edges, 1, np.pi / 180, 20)
        # Add logic to process lines

    return False


def match_template_multi_scale_angle(image, template, scales, angles, best_match, method=cv2.TM_CCOEFF_NORMED):
    """
    Matches a template in an image at multiple scales and angles.

    Args:
        image (numpy.ndarray): The input image in which to search for the template.
        template (numpy.ndarray): The template image to search for.
        scales (list of float): List of scales to resize the template.
        angles (list of float): List of angles to rotate the template.
        method (int): Matching method to use. Default is cv2.TM_CCOEFF_NORMED.

    Returns:
        dict: Dictionary containing the best match's location, scale, angle, and match value.
    """

    # template中的黑色部分不参与匹配
    _, mask = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # mask = cv2.bitwise_not(mask)
    # if debug:
        # cv2.imshow("Template", template)
        # cv2.imshow("Image", image)
        # cv2.imshow("Mask", mask)
        # cv2.waitKey(0)

    for scale in scales:
        print("scale:", scale)
        for angle in angles:
            # Resize the template
            scaled_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
            scaled_mask = cv2.resize(mask, (0, 0), fx=scale, fy=scale)
            # if template larger than image, skip
            if scaled_template.shape[0] > image.shape[0] or scaled_template.shape[1] > image.shape[1]:
                continue
            h, w = scaled_template.shape[:2]
            # Rotate the template
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_template = cv2.warpAffine(scaled_template, rotation_matrix, (w, h))
            rotated_mask = cv2.warpAffine(scaled_mask, rotation_matrix, (w, h))
            # if debug:
                # cv2.imshow("Rotated Template", rotated_template)
                # cv2.imshow("Rotated Mask", rotated_mask)
                # cv2.waitKey(0)

            # Match the template
            # result = cv2.matchTemplate(image, rotated_template, method, mask=rotated_mask)
            result = cv2.matchTemplate(image, rotated_template, method)

            # result = cv2.matchTemplate(image, rotated_template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            # check if max_val is inf
            if np.isinf(max_val):
                continue
            if debug and True:
                # concat image and template
                h, w = rotated_template.shape[:2]
                image_show = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                image_show[0:h, 0:w] = cv2.cvtColor(rotated_template, cv2.COLOR_GRAY2BGR)
                # draw rect and max_val
                cv2.rectangle(image_show, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 0, 255), 2)
                cv2.putText(image_show, f"{round(max_val,3)}", (max_loc[0], max_loc[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow("Result ", image_show)
                cv2.waitKey(0) 

            # Check if this match is better than the best match so far
            if max_val > best_match["max_val"]:
                print("got better match")
                best_match.update({
                    "max_val": max_val,
                    "max_loc": max_loc,
                    "scale": scale,
                    "angle": angle,
                    "template": image_show,
                })

    return best_match


def preprocess(img):
    img_blur = cv2.GaussianBlur(img, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 50, 50)
    kernel = np.ones((3, 3))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=2)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    return img_erode

def find_tip(points, convex_hull):
    length = len(points)
    indices = np.setdiff1d(range(length), convex_hull)

    for i in range(2):
        j = indices[i] + 2
        if j > length - 1:
            j = length - j
        if np.all(points[j] == points[indices[i - 1] - 2]):
            return tuple(points[j])



if __name__ == '__main__':
    image_folder = './image'
    destination_text = "输液区"
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
                # 通过 levenshtein 距离来进行模糊匹配，防止输入的目标文本有误差
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


                # top_left, bottom_right = text_target.coordinates.bounding_box()
                # extend_pixel = (bottom_right[0] - top_left[0]) * 2
                # top_left = (max(0, top_left[0] - extend_pixel), max(0, top_left[1] - extend_pixel))
                # bottom_right = (min(image.shape[1], bottom_right[0] + extend_pixel), min(image.shape[0], bottom_right[1] + extend_pixel))
                # roi = mask_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                # # resize 2x larger
                # roi = cv2.resize(roi, (0, 0), fx=2, fy=2)
                
                # _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # contours, hierarchy = cv2.findContours(preprocess(roi), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                # imgShow = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                # # draw all contours
                # cv2.drawContours(imgShow, contours, -1, (255, 0, 0), 1)
                # if debug:
                #     cv2.imshow("Contours", imgShow)
                #     cv2.waitKey(0)

                # for cnt in contours:
                #     peri = cv2.arcLength(cnt, True)
                #     approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
                #     hull = cv2.convexHull(approx, returnPoints=False)
                #     sides = len(hull)
                #     # draw current cnt

                #     if 6 > sides > 3 and sides + 2 == len(approx):
                #         print("Found arrow")
                #         arrow_tip = find_tip(approx[:,0,:], hull.squeeze())
                #         cv2.drawContours(imgShow, [cnt], -1, (0, 0, 255), 1)
                #         # show info
                #         if arrow_tip:
                #             cv2.drawContours(imgShow, [cnt], -1, (0, 255, 0), 3)
                #             cv2.circle(imgShow, arrow_tip, 3, (0, 255, 0), cv2.FILLED)
                # cv2.imshow("Arrow", imgShow)
                # cv2.waitKey(0)

                if True:
                    # detect_arrow(mask_image, text_target, debug)
                    # 模板匹配
                    template = cv2.imread("./template/image.png", cv2.IMREAD_GRAYSCALE)
                    # 缩小2x
                    template = cv2.resize(template, (0, 0), fx=0.5, fy=0.5)
                    # 模板预处理，二值化并留下黑色部分
                    _, template = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    template = cv2.bitwise_not(template)
                    # template 逆时针旋转90度 构建新的模板
                    template_list = [template]
                    template = cv2.rotate(template, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    template_list.append(template)
                    template = cv2.rotate(template, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    template_list.append(template)



                    # scales = [0.5, 0.75, 1.0]
                    scales = [0.75, 1.0, 1.5]
                    angles = [-15, 0, 15]
                    top_left, bottom_right = text_target.coordinates.bounding_box()
                    extend_pixel = (bottom_right[0] - top_left[0]) * 2
                    top_left = (max(0, top_left[0] - extend_pixel), max(0, top_left[1] - extend_pixel))
                    bottom_right = (min(image.shape[1], bottom_right[0] + extend_pixel), min(image.shape[0], bottom_right[1] + extend_pixel))
                    roi = mask_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                    # OTSU
                    # roi = cv2.GaussianBlur(roi, (5, 5), 0)
                    _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    if debug:
                        cv2.imshow("ROI", roi)
                        cv2.waitKey(0)

                    best_match = {
                        "max_val": 0.0,
                        "max_loc": None,
                        "scale": None,
                        "angle": None,
                        "template":None,
                        }

                    for template in template_list:
                        best_match = match_template_multi_scale_angle(roi, template, scales, angles, best_match)

                    if best_match["max_val"] < 0.4 :
                        print("Cannot find the target image")
                        continue
                    print(f"Best match at location {best_match['max_loc']} with scale {best_match['scale']} and angle {best_match['angle']}")
                    print(f"Match value: {best_match['max_val']}")

                    # if best_match['max_val'] > 0.3:
                    # show best_match['template']
                    cv2.imshow("Best Match", best_match['template'])
                    cv2.waitKey(0)
                    # Draw rectangle around the best match
                    h, w = best_match['template'].shape[:2]
                    best_scale = best_match["scale"]
                    best_angle = best_match["angle"]

                    # Resize and rotate template to best match scale and angle
                    scaled_template = cv2.resize(best_match['template'], (0, 0), fx=best_scale, fy=best_scale)
                    h_scaled, w_scaled = scaled_template.shape[:2]
                    # center = (w_scaled // 2, h_scaled // 2)
                    # rotation_matrix = cv2.getRotationMatrix2D(center, best_angle, 1.0)
                    # rotated_template = cv2.warpAffine(scaled_template, rotation_matrix, (w_scaled, h_scaled))

                    top_left = best_match["max_loc"]
                    bottom_right = (top_left[0] + w_scaled, top_left[1] + h_scaled)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            else:
                print(f"Cannot find the target text '{destination_text}'")

            print(f"OCR Execution time: {round(execution_time * 1e3, 2)} ms")
