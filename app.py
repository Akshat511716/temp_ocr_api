from collections import Counter
from difflib import SequenceMatcher

import cv2
import easyocr
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, request

app = Flask(__name__)


@app.route('/')
def home():
    return "<h1>Working properly</h1>"


def get_image(file):
    # OCR using Image Processing
    # Image Preprocessing

    # 3. Deskew
    # READ INPUT IMAGE
    pil_image = file.convert('RGB')
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    img = open_cv_image

    # 3. Deskew

    image_file = img
    from scipy.ndimage import rotate

    def correct_skew(image, delta=1, limit=5):
        def determine_score(arr, angle):
            data = rotate(arr, angle, reshape=False, order=0)
            histogram = np.sum(data, axis=1, dtype=float)
            score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
            return histogram, score

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        scores = []
        angles = np.arange(-limit, limit + delta, delta)
        for angle in angles:
            histogram, score = determine_score(thresh, angle)
            scores.append(score)

        best_angle = angles[scores.index(max(scores))]

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)

        return best_angle, corrected

    _, corrected = correct_skew(image_file)
    image_file = [corrected]

    # Easy OCR
    texts = []
    text_reader = easyocr.Reader(['en'])  # Initializing the ocr

    for image in image_file:
        result = text_reader.readtext(image)
        text_result = ""
        for i in result:
            text_result += i[1] + " "
        texts.append(text_result.strip())


    new_list = [x for x in texts if x != '']
    texts = new_list

    # Use Spell check on the OCR results
    from spellchecker import SpellChecker
    spell = SpellChecker()
    temp_list = []
    for i in texts:
        temp = ""
        for j in i.split(" "):
            if spell.correction(j) is None:
                temp += j + " "
            else:
                temp += str(spell.correction(j)) + " "
        temp_list.append(temp)
    texts = temp_list

    # Remove meaningless word using english words dataset
    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

    temp_list = []
    word_data = pd.read_csv('unigram_freq.csv')
    word_data = list(word_data['word'])
    for i in texts:
        temp = ""
        for j in i.split():
            for k in word_data:
                if similar(j, str(k)) > 0.6 and len(j) > 1:
                    temp += j + " "
                    break
        temp_list.append(temp.strip())
    texts = temp_list

    # Correct the grammar of the results
    temp_list = []
    from gingerit.gingerit import GingerIt
    for i in texts:
        corrected_text = GingerIt().parse(i)
        temp_list.append(corrected_text['result'])
    texts = temp_list

    # Remove empty elements
    temp_list = []
    for i in texts:
        temp = ""
        for j in i:
            if j.isalnum() or j == ' ':
                temp += j
        temp_list.append(temp.strip())
    texts = temp_list

    # Max of 6 words in OCR results
    # Number of words to consider in results of OCR
    words = 6
    ocr_result = []
    for i in texts:
        book = ""
        for i in i.split(' ')[:words]:
            book += i + " "
        book = book.strip()
        ocr_result.append(book)

    for i in ocr_result:
        if i == "" or i == " ":
            ocr_result.remove(i)

    def max_occurring_element(lst):
        counts = Counter(lst)
        max_count = max(counts.values())
        max_elements = [k for k, v in counts.items() if v == max_count]
        if max_elements:
            return max_elements[0]
        else:
            return lst[0]

    # store(username, str(most_frequent(ocr_result)))
    return f"{max_occurring_element(ocr_result)}"


@app.route('/upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    # data = request.form.to_dict()

    # check if the file is empty
    if file.filename == '':
        return 'No selected file'

    # save the file
    # file.save("temp/" + file.filename)
    # img = Image.open(file)
    # Get the username from the data
    # username = data.get('username', '')
    img = Image.open(file.stream)
    result = get_image(img)
    return f"{result}"


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
    # app.run(debug=True)
