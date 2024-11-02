import os

from ultralytics import YOLO
import numpy as np
from PIL import Image

model = YOLO(r'runs\detect\train3\weights\best.pt')
results = model(r'split\images\test', save=True, save_txt=True)

# Show the results
# i = 0
# for r in results:
#     im_array = r.plot()  # plot a BGR numpy array of predictions
#     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
#     # im.show()  # show image
#     # print(r.path)
#     image_name = os.path.basename(r.path)
#
#     # print(image_name)
#     im.save(os.path.join(r'D:\python_project\ultralytics-main\predict\images', f'{image_name}')) # save image
    # i+=1