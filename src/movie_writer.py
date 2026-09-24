#!/usr/bin/env python
"""Encoding the frames Iseult draws into a movie file with ffmpeg.

Frames are rendered straight to raw RGBA and piped to ffmpeg, which encodes
them as H.264. The quality is set by CRF below: 18 is about where x264's
compression stops being visible, so the movie looks like the figure on screen
while being a small fraction of the size of the lossless 10-bit ProRes 4444
Iseult used to write.
"""
import io
import subprocess

import numpy as np

# x264's constant rate factor: 0 is lossless, ffmpeg's default is 23 and each
# +6 roughly halves the file size. Lower it for better quality.
CRF = 18


def render_frame(figure, dpi=None):
    '''Draw `figure` at `dpi` (default: the figure's own) as an (h, w, 4) uint8 RGBA array.'''
    if dpi is None or dpi == figure.dpi:
        # The figure on screen is already drawn at this resolution; reuse
        # that rather than draw it all again. Redraw only if something has
        # changed since, e.g. a legend being put back where the user left it.
        if figure.stale:
            figure.canvas.draw()
        return np.asarray(figure.canvas.buffer_rgba()).copy()
    buf = io.BytesIO()
    figure.savefig(buf, format='rgba', dpi=dpi, facecolor=figure.get_facecolor())
    # This is how Agg sizes its canvas: the figure size in pixels, truncated.
    width, height = (int(v) for v in figure.get_size_inches() * dpi)
    return np.frombuffer(buf.getbuffer(), dtype=np.uint8).reshape(height, width, 4)


class MovieWriter:
    '''Pipe RGBA frames into an H.264 movie.

    Use as a context manager, or call close() when done. ffmpeg is started on
    the first frame, whose size every later frame must match.'''

    def __init__(self, path, fps):
        self.path = str(path)
        self.fps = fps
        self.frame_shape = None
        self._proc = None
        self._cmd = None

    def _start(self, height, width):
        self._cmd = ['ffmpeg', '-y', '-loglevel', 'error',
                     '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{width}x{height}',
                     '-framerate', str(int(self.fps)), '-i', '-',
                     # 4:2:0 video needs an even width and height; drop the odd
                     # last row or column rather than stretch the frame.
                     '-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2:0:0,'
                            'scale=out_color_matrix=bt709:out_range=tv',
                     '-c:v', 'libx264', '-preset', 'medium', '-crf', str(CRF),
                     # yuv420p rather than yuv444p so that the movie also plays
                     # in QuickTime, PowerPoint, Keynote and browsers.
                     '-pix_fmt', 'yuv420p',
                     '-colorspace', 'bt709', '-color_primaries', 'bt709',
                     '-color_trc', 'bt709', '-color_range', 'tv',
                     '-movflags', '+faststart',
                     self.path]
        try:
            self._proc = subprocess.Popen(self._cmd, stdin=subprocess.PIPE)
        except FileNotFoundError:
            raise RuntimeError('Making a movie needs ffmpeg, which was not found on the PATH.') from None

    def write(self, frame):
        '''Add an (h, w, 4) uint8 RGBA frame.'''
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 4:
            raise ValueError(f'Expected an (h, w, 4) RGBA frame, got shape {frame.shape}')
        if self._proc is None:
            self.frame_shape = frame.shape
            self._start(*frame.shape[:2])
        elif frame.shape != self.frame_shape:
            raise ValueError(f'Frame size changed from {self.frame_shape[1]}x{self.frame_shape[0]} '
                             f'to {frame.shape[1]}x{frame.shape[0]} part way through the movie')
        self._proc.stdin.write(frame.data)

    def close(self):
        if self._proc is None:
            return
        self._proc.stdin.close()
        self._proc.wait()
        if self._proc.returncode != 0:
            raise subprocess.CalledProcessError(self._proc.returncode, self._cmd)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        elif self._proc is not None:
            # Don't leave ffmpeg waiting on a pipe that will never be written to.
            self._proc.stdin.close()
            self._proc.wait()
        return False
