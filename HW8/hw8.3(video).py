import cv2
import numpy as np

# define a video capture object
vid = cv2.VideoCapture(0)
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

writer = cv2.VideoWriter('outputVideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))
while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    writer.write(frame)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
writer.release()
# Destroy all the windows
cv2.destroyAllWindows()