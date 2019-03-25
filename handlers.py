from PIL import Image, ImageTk
import cv2
import time

from filters import *


_default_filter_map = {
    'c': (clarendon, 'Clarendon'),
    'k': (kelvin, 'Kelvin'),
    'm': (moon, 'Moon'),
    'x': (xpro2, 'Xpro2'),
    'o': (cartoon, 'Cartoon'),
    'b': (sketch_pencil_using_blending, 'Sketch pencil using blending'),
    'e': (sketch_pencil_using_edge_detection, 'Sketch pencil using edge detection'),
    'i': (invert, 'Invert'),
    'w': (black_and_white, 'Black annd white'),
    'r': (warming, 'Warming'),
    't': (cooling, 'Cooling'),
    'a': (cartoon2, 'Cartoon2'),
    'n': (no_filter, 'No Filter')
}


class RootHandler:

    def __init__(self, panel):
        self.panel = panel
        self.curr_func = no_filter

    def bind_root(self, root, img_handler, init=True):
        for ch, (fn, name) in _default_filter_map.items():
            print(f'Press {ch} - {name}')
            root.bind(f'<{ch}>', lambda e, fn=fn: fn(self.panel, img_handler, self, e, init))

        print(f'\nPress ESC to quit...\n')

    def update_func(self, ch):
        self.curr_func = _default_filter_map[ch][0]

    def call_func(self, img_handler):
        self.curr_func(self.panel, img_handler)


class ImageHandler:

    def __init__(self, frame, filtered_frame, out_path):
        self.frame = frame
        self.filtered_frame = filtered_frame
        self.out_path = out_path

    def update_label(self, label, img, orig=False):
        if orig:
            self.frame = img
        else:
            self.filtered_frame = img

        if img is not None:
            img = Image.fromarray(img[..., ::-1])
            img = ImageTk.PhotoImage(img)
            label.configure(image=img)
            label.image = img

    def save_img(self, orig=False):
        ts = time.localtime(time.time())
        self.out_path.mkdir(parents=True, exist_ok=True)

        if orig:
            filename = f'orig-{time.strftime("%Y-%m-%d_%H-%M-%S", ts)}.jpg'
            p = self.out_path / filename
            cv2.imwrite(str(p), self.frame.copy())
        else:
            filename = f'filter-{time.strftime("%Y-%m-%d_%H-%M-%S", ts)}.jpg'
            p = self.out_path / filename
            cv2.imwrite(str(p), self.filtered_frame.copy())

        print(f'[INFO] saved {filename}')
