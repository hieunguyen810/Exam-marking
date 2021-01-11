import cv2
import numpy as np
import image_processing
import remove
import matplotlib.pyplot as plt 
mser = cv2.MSER_create()
def non_max_suppression(boxes, overlapThresh):
  if len(boxes)==0:
    return []
  if boxes.dtype.kind == "i":
    boxes = boxes.astype("float")
  pick = []

  x1 = boxes[:,0]
  y1 = boxes[:,1]
  x2 = boxes[:,2]
  y2 = boxes[:,3]

  area = (x2 - x1 + 1) * (y2 - y1 + 1)
  idxs = np.argsort(y2)
    # Khởi tạo một vòng while loop qua các index xuất hiện trong indexes
  while len(idxs) > 0:
    # Lấy ra index cuối cùng của list các indexes và thêm giá trị index vào danh sách các indexes được lựa chọn
    last = len(idxs) - 1
    i = idxs[last]
    pick.append(i)

    # Tìm cặp tọa độ lớn nhất (x, y) là điểm bắt đầu của bounding box và tọa độ nhỏ nhất (x, y) là điểm kết thúc của bounding box
    xx1 = np.maximum(x1[i], x1[idxs[:last]])
    yy1 = np.maximum(y1[i], y1[idxs[:last]])
    xx2 = np.minimum(x2[i], x2[idxs[:last]])
    yy2 = np.minimum(y2[i], y2[idxs[:last]])

    # Tính toán width và height của bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    # Tính toán tỷ lệ diện tích overlap
    overlap = (w * h) / area[idxs[:last]]

    # Xóa index cuối cùng và index của bounding box mà tỷ lệ diện tích overlap > overlapThreshold
    idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
  # Trả ra list các index được lựa chọn
  return boxes[pick].astype("int")
def _drawBoundingBox(img, cnt):
  x,y,w,h = cv2.boundingRect(cnt)
  img = cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
  return img
def detect_bounding_box(image):
  regions, _ = mser.detectRegions(image)
  hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
  boundingBoxes = []
  imgOrigin = image.copy()
  area_cnt = [cv2.contourArea(cnt) for cnt in regions]
  area_sort = np.argsort(area_cnt)[::-1]
  for i in area_sort[:15]:
      cnt = hulls[i]
      imgOrigin = _drawBoundingBox(imgOrigin, cnt)
      x,y,w,h = cv2.boundingRect(cnt)
      x1, y1, x2, y2 = x, y, x+w, y+h
      boundingBoxes.append((x1, y1, x2, y2))
  boundingBoxes = np.array(boundingBoxes)
  pick = non_max_suppression(boundingBoxes, 0.5)
  num = 1
  for (startX, startY, endX, endY) in pick:
      imgOrigin = cv2.rectangle(imgOrigin, (startX, startY), (endX, endY), (0, 255, 0), 2)
      crop = imgOrigin[startY:endY, startX:endX]
      file_name = "./Box/box%s.png" % num
      cv2.imwrite(file_name, crop)
      num+=1
  # remove grid
  img = cv2.imread("./Box/box1.png", 0)
  img = cv2.threshold(img, 0,255, cv2.THRESH_OTSU)[1]
  img = cv2.dilate(img, (10, 10))
  cv2.imwrite("./Box/box11.png", img)
def detect_letter():
  # Load image, grayscale
  image = cv2.imread('./Box/box11.png')
  # image = image_processing.remove_non_text(image)
  image = cv2.resize(image, (1000, 500))
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray = image_processing.erode(gray)

  regions, _ = mser.detectRegions(gray)
  hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

  area_cnt = [cv2.contourArea(cnt) for cnt in regions]
  area_sort = np.argsort(area_cnt)[::-1]
  boundingBoxes = []
  imgOrigin = image.copy()
  for i in area_sort[:600]:
      cnt = hulls[i]
      imgOrigin = _drawBoundingBox(imgOrigin, cnt)
      x,y,w,h = cv2.boundingRect(cnt)
      x1, y1, x2, y2 = x, y, x+w, y+h
      boundingBoxes.append((x1, y1, x2, y2))
  #plt.imshow(imgOrigin)
  #plt.show()

  # Remove đi bounding box parent (chính là khung hình bound toàn bộ hình ảnh), nếu không khi áp dụng non max suppression chỉ giữ lại bounding box này
  boundingBoxes = [box for box in boundingBoxes if box[:2] != (0, 0)]
  boundingBoxes = np.array(boundingBoxes)
  pick = non_max_suppression(boundingBoxes, 0.5)
  print(pick)
  a = []
  b = []
  c = []
  d = []
  e = []
  for i in np.arange(len(pick)):
    if pick[i][0] < 200:
      a = np.append(a, pick[i])
    elif pick[i][0] > 200 and pick[i][0] < 400:
      b = np.append(b, pick[i])
    elif pick[i][0] > 400 and pick[i][0] < 600:
      c = np.append(c, pick[i])
    elif pick[i][0] > 600 and pick[i][0] < 800:
      d = np.append(d, pick[i])
    elif pick[i][0] > 800:
      e = np.append(e, pick[i])
  a = a.reshape(int(len(a)/4), 4)
  b = b.reshape(int(len(b)/4), 4)
  c = c.reshape(int(len(c)/4), 4)
  d = d.reshape(int(len(d)/4), 4)
  e = e.reshape(int(len(e)/4), 4)
  a = a[a[:,1].argsort()].astype(int)
  b = b[b[:,1].argsort()].astype(int)
  c = c[c[:,1].argsort()].astype(int)
  d = d[d[:,1].argsort()].astype(int)
  e = e[e[:,1].argsort()].astype(int)
  pick = np.vstack([a, b, c, d, e])
  num = 1
  for (startX, startY, endX, endY) in pick:
      imgOrigin = cv2.rectangle(imgOrigin, (startX, startY), (endX, endY), (0, 255, 0), 2)
      crop = imgOrigin[startY:endY, startX:endX]
      file_name = "./Letter_extracted/%s.png" % num
      cv2.imwrite(file_name, crop)
      num+=1
  remove.remove_deleted_image()
  

