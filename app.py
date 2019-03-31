import cv2
import numpy as np
from model import NeuralNet

net = NeuralNet()


canvas = np.ones((600,600), dtype="uint8") * 255

canvas[100:500,100:500] = 0

start_point = None
end_point = None
is_drawing = False

def draw_line(img,start_at,end_at):
    cv2.line(img,start_at,end_at,255,10)

def on_mouse_events(event,x,y,flags,params):
    global start_point
    global end_point
    global canvas
    global is_drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing=True
        if is_drawing:
            start_point = (x,y)
    elif event == cv2.EVENT_MOUSEMOVE:
        
        if is_drawing:
            end_point = (x,y)
            draw_line(canvas,start_point,end_point)
            start_point = end_point
    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False


cv2.namedWindow("Testing Board")
cv2.setMouseCallback("Testing Board", on_mouse_events)


while(True):
    cv2.imshow("Testing Board", canvas)
    key = cv2.waitKey(1) & 0xFF 
    if key == ord('q'):
        break
   
    elif key == ord('c'):
        canvas[100:500,100:500] = 0
    elif key == ord('p'):
        image = canvas[100:500,100:500]
        result = net.predict(image)
        print("PREDICTION : ",result)

cv2.destroyAllWindows()
