import cv2

#Quelle: https://www.programcreek.com/python/example/89439/cv2.pointPolygonTest
#angepasst mit reduziertem Polygon

def mark_hand_center(frame_in, cont,appox):
    max_d = 0
    pt = (0, 0)
    epsilon = appox * cv2.arcLength(cont, True)
    polyCont = cv2.approxPolyDP(cont, epsilon, True)
    x, y, w, h = cv2.boundingRect(polyCont)
    for ind_y in range(int(y + 0.3 * h), int(y + 0.8 * h)):  # around 0.25 to 0.6 region of height (Faster calculation with ok results)
        for ind_x in range(int(x + 0.3 * w), int(x + 0.6 * w)):  # around 0.3 to 0.6 region of width (Faster calculation with ok results)
            dist = cv2.pointPolygonTest(polyCont, (ind_x, ind_y), True)
            if (dist > max_d):
                max_d = dist
                pt = (ind_x, ind_y)
    if (max_d > frame_in.shape[1]):
        thresh_score = True
        cv2.circle(frame_in, pt, int(max_d), (255, 0, 0), 2)
    else:
        thresh_score = False
    return frame_in, pt, max_d, thresh_score