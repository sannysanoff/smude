import argparse
import os
import shlex
import subprocess
from typing import Dict, Optional

from skimage.io import imread

from . import Smude


class SmudeTUI:
    def __init__(self, infile: str, params: Dict):
        self.infile = infile
        self.params = params
        self.smude = Smude(
            use_gpu=params.get('use_gpu', False),
            binarize_output=params.get('binarize_output', True),
            verbose=True,
            noise_reduction=params.get('noise_reduction'),
            max_dist=params.get('max_dist', 40.0),
            threshold=params.get('threshold', 128),
            sauvola_k=params.get('sauvola_k', 0.25),
            skip_border_removal=params.get('skip_border_removal', False),
            grow=params.get('grow', 0),
            spline_threshold=params.get('spline_threshold', 80),
            pad=params.get('pad', 0),
            roi_threshold=params.get('roi_threshold', 0.4),
        )
        self.image = imread(self.infile)
        self.stage_files = []
        self.stage_index = 0
        self._last_result = None

    def run_pipeline(self):
        result = self.smude.process(self.image)
        self._last_result = result
        self._collect_verbose_images()

    def _collect_verbose_images(self):
        patterns = [
            'verbose_input.jpg',
            'verbose_roi_mask.jpg',
            'verbose_masked_roi.jpg',
            'verbose_full_image_no_roi.jpg',
            'verbose_before_enhance.jpg',
            'verbose_01_mask.jpg',
            'verbose_02_gray_normalized.jpg',
            'verbose_03_blurred.jpg',
            'verbose_04_contrast_enhanced.jpg',
            'verbose_05_binary.jpg',
            'verbose_06_final_rgb.jpg',
            'verbose_after_flood_fill.jpg',
            'verbose_binarized.jpg',
            'verbose_padded.jpg',
            'verbose_preprocessed_for_unet.jpg',
            'verbose_unet_background.jpg',
            'verbose_unet_upper.jpg',
            'verbose_unet_lower.jpg',
            'verbose_unet_barlines.jpg',
            'verbose_final_result.jpg',
        ]
        self.stage_files = [p for p in patterns if os.path.exists(p)]
        if not self.stage_files and os.path.exists(self.infile):
            self.stage_files = [self.infile]
        self.stage_index = 0

    def imgcat(self, path: str):
        if not os.path.exists(path):
            return
        cmd = f"imgcat {shlex.quote(path)}"
        try:
            subprocess.run(cmd, shell=True, check=False)
        except Exception:
            pass

    def render(self):
        os.system('clear')
        if not self.stage_files:
            print('No images to display')
            return
        current = self.stage_files[self.stage_index]
        print(f"[Smude TUI] Stage {self.stage_index+1}/{len(self.stage_files)}: {os.path.basename(current)}")
        print("Controls: space/Right=next, Left=prev, Enter=next-phase, +/- adjust threshold 10%, k/K adjust sauvola_k ±10%, d/D max_dist ±10%, g/G grow ±1, s toggle skip_border_removal, r rerun, q quit")
        self.imgcat(current)

    def adjust(self, key: str):
        def inc(val, pct=0.1):
            return type(val)(val + (val * pct))
        def dec(val, pct=0.1):
            return type(val)(val - (val * pct))

        if key == '+':
            self.smude.threshold = int(max(0, min(255, inc(self.smude.threshold))))
        elif key == '-':
            self.smude.threshold = int(max(0, min(255, dec(self.smude.threshold))))
        elif key == 'k':
            self.smude.sauvola_k = max(0.0, dec(self.smude.sauvola_k))
        elif key == 'K':
            self.smude.sauvola_k = dec(self.smude.sauvola_k, -0.1)
        elif key == 'd':
            self.smude.max_dist = max(1.0, dec(self.smude.max_dist))
        elif key == 'D':
            self.smude.max_dist = inc(self.smude.max_dist)
        elif key == 'g':
            self.smude.grow = max(0, self.smude.grow - 1)
        elif key == 'G':
            self.smude.grow = self.smude.grow + 1
        elif key == 's':
            self.smude.skip_border_removal = not self.smude.skip_border_removal

    def next_image(self):
        if self.stage_index < len(self.stage_files) - 1:
            self.stage_index += 1
    def prev_image(self):
        if self.stage_index > 0:
            self.stage_index -= 1

    def loop(self):
        self.run_pipeline()
        self.render()
        import termios, tty, sys
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch == 'q':
                    break
                elif ch in (' ', '\x1b[C'):
                    self.next_image()
                    self.render()
                elif ch in ('\x1b[D',):
                    self.prev_image()
                    self.render()
                elif ch == '\r':
                    self.next_image()
                    self.render()
                elif ch in ['+', '-', 'k', 'K', 'd', 'D', 'g', 'G', 's']:
                    self.adjust(ch)
                    self.run_pipeline()
                    self.render()
                elif ch == 'r':
                    self.run_pipeline()
                    self.render()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main(argv: Optional[list] = None):
    parser = argparse.ArgumentParser(description='Smude interactive TUI with kitty graphics')
    parser.add_argument('infile')
    parser.add_argument('--use-gpu', action='store_true')
    args = parser.parse_args(argv)
    params = {
        'use_gpu': args.use_gpu,
    }
    app = SmudeTUI(args.infile, params)
    app.loop()


if __name__ == '__main__':
    main()
