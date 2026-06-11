import pdfplumber
import cv2
import numpy as np

with pdfplumber.open('C:/Users/EXAVALU/Downloads/Sample filled Claim Form.pdf') as pdf:
    for page_num in [1, 3]:
        page = pdf.pages[page_num]
        print(f'Page {page_num+1} has {len(page.images)} images')
        for i, img_obj in enumerate(page.images[:5]):
            bbox = (img_obj['x0'], img_obj['top'], img_obj['x1'], img_obj['bottom'])
            try:
                cropped = page.crop(bbox).to_image(resolution=200).original
                cv_img = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                variance = cv2.Laplacian(gray, cv2.CV_64F).var()
                w = img_obj.get('width')
                h = img_obj.get('height')
                print(f'Image {i} variance: {variance}, size: {w}x{h}')
            except Exception as e:
                print(e)
