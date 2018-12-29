from PIL import Image, ImageTk
import threading
import tkinter as tk
from imutils.video import VideoStream
import cv2
import argparse
from pathlib import Path
import time

from handlers import ImageHandler, RootHandler


class PhotoFilter:

    def __init__(self, vs, out_path, orig=True):
        self.vs = vs
        self.out_path = out_path
        self.thread = None
        self.stop_event = None

        self.init = True
        self.root = tk.Tk()
        self.root.title('Video Filter')
        self.orig = None
        self.panel = None
        self.img_handler = None
        self.root_handler = None

        btn_o = tk.Button(self.root, text='Snapshot', bd=5,
                          command=lambda: self.img_handler.save_img(True))
        btn_p = tk.Button(self.root, text='Snapshot', bd=5,
                          command=lambda: self.img_handler.save_img())

        btn_o.grid(row=1, column=0)
        btn_p.grid(row=1, column=1)

        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._video_loop, args=())
        self.thread.start()

        self.root.wm_protocol('WM_DELETE_WINDOW', self._on_close)

    def _video_loop(self):
        try:
            while not self.stop_event.is_set():
                frame = self.vs.read()

                if self.init:
                    self.init = False
                    image = Image.fromarray(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    image = ImageTk.PhotoImage(image)

                    self.orig = tk.Label(self.root, image=image)
                    self.orig.grid(row=0, column=0)
                    self.panel = tk.Label(self.root, image=image)
                    self.panel.grid(row=0, column=1)

                    self.img_handler = ImageHandler(
                        frame, frame, self.out_path)
                    self.root_handler = RootHandler(self.panel)

                    self.root_handler.bind_root(
                        self.root, self.img_handler, init=False)
                    self.root.bind('<Escape>', lambda e: self._on_close())

                else:
                    self.img_handler.update_label(self.orig, frame, orig=True)
                    self.root_handler.call_func(self.img_handler)

        except RuntimeError as e:
            print('[INFO] Caught a RuntimeError')

    def _on_close(self):
        print('[INFO] closing...')
        self.stop_event.set()
        self.vs.stop()
        self.root.destroy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments for video filters')
    parser.add_argument('-o', '--out_path', type=str,
                        default='./snapshots', help='Path to directory to store snapshots')
    parser.add_argument('-p', '--picamera', type=int, default=0,
                        help='Wheather or not the Rasberry Pi camera should be used')
    args = parser.parse_args()

    print('[INFO] warming up camera')
    vs = VideoStream(usePiCamera=args.picamera > 0).start()
    time.sleep(2.0)

    out_path = Path(args.out_path)

    pb = PhotoFilter(vs, out_path)
    pb.root.mainloop()
