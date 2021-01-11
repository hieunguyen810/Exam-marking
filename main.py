from tkinter import *
from tkinter import filedialog
import pytesseract as tess
from PIL import Image, ImageEnhance, ImageFilter, ImageTk, ImageOps
#tess.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
import image_processing
import pandas as pd
import numpy as np
import cv2
from sklearn import metrics
import recognition
import detection
import remove
import digits_recognizing
root = Tk()
def choose_image():
    global image_path
    image_path = filedialog.askopenfilename()
def choose_csv():
    global csv_path
    csv_path = filedialog.askopenfilename()
def new_photo():
    label3 = Label(root, text="Please take a new photo", fg="red", font=("Helvetica", 13))
    label3.pack()
def marking():
    global csv_path
    global image_path
    img = cv2.imread(image_path, 0)
    answer = pd.read_csv(csv_path, header = None)
    answer = answer.values
    answer = np.array_str(answer)
    answer = answer.lower()
    output = []
    for i in answer:
        number = ord(i) - 96
        output.append(number)
    u = []
    for i in output:
        if i > 0:
            u = np.append(u, i)
    remove.clear_folder("./Letter")
    remove.clear_folder("./Box")
    remove.clear_folder("./Digits")
    remove.clear_folder("./Letter_extracted")
    detection.detect_bounding_box(img)
    # detect error
    box = cv2.imread("./Box/box1.png", 0)
    a, b = box.shape
    if a*b < 150000:
        new_photo()
    try: 
        box_id = cv2.imread("Box/box2.png", 0)
    except:
        new_photo()
    student_id = digits_recognizing.recognition('./Box/box2.png')
    label2 = Label(root, text=student_id, fg="blue", font=("Helvetica", 13))
    label2.pack()

    text = recognition.image_to_string()
    u = u - 1
    print(u)
    print(text)
    score = metrics.accuracy_score(text, u)
    score = float("{:.3f}".format(score))
    print("Your score: ", score)

    label1 = Label(root, text=score*10, fg="red", font=("Helvetica", 16))
    label1.pack()
def main():
    root.title("Exam marking")
    root.geometry('500x350')
    image_path = ""
    csv_path = ""

    background_image = Image.open('background.jpg')
    background_image = background_image.resize((500, 350))
    background_image = ImageTk.PhotoImage(background_image)
    background_label = Label(root, image=background_image)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    Choose_image_button = Button(root, text = "Choose an image", command = choose_image)
    Choose_image_button.place(x = 100, y = 50)
    
    Choose_csv_button = Button(root, text = "Choose an answer file", command = choose_csv)
    Choose_csv_button.place(x = 300, y = 50)

    Marking_button = Button(root, text = "Marking", command = marking)
    Marking_button.place(x= 230, y = 200)

    root.mainloop()
if __name__ == '__main__':
    main()

