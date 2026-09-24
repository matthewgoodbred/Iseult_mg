import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

import movie_writer


@pytest.fixture
def figure():
    # A size that is an odd number of pixels at most dpis
    fig, ax = plt.subplots(figsize=(3.37, 2.11), dpi=100)
    ax.imshow(np.random.default_rng(0).random((20, 30)))
    yield fig
    plt.close(fig)


@pytest.mark.parametrize('dpi', [None, 100, 72, 157.3])
def test_render_frame_matches_savefig(figure, dpi, tmp_path):
    frame = movie_writer.render_frame(figure, dpi)
    figure.savefig(tmp_path / 'ref.png', dpi=figure.dpi if dpi is None else dpi)
    reference = plt.imread(tmp_path / 'ref.png')
    assert frame.dtype == np.uint8
    assert frame.shape == reference.shape
    np.testing.assert_array_equal(frame, np.round(reference*255).astype(np.uint8))


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='needs ffmpeg')
def test_writes_h264_movie(figure, tmp_path):
    path = tmp_path / 'test.mov'
    with movie_writer.MovieWriter(path, fps=5) as movie:
        for _ in range(3):
            movie.write(movie_writer.render_frame(figure, 100))
    assert path.stat().st_size > 0
    if shutil.which('ffprobe'):
        info = subprocess.check_output(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
             '-show_entries', 'stream=codec_name,pix_fmt,width,height,nb_read_frames',
             '-of', 'default=noprint_wrappers=1', str(path)], text=True)
        fields = dict(line.split('=') for line in info.split())
        assert fields['codec_name'] == 'h264'
        assert fields['pix_fmt'] == 'yuv420p'
        # 337 x 211 pixels, cropped to even dimensions
        assert (fields['width'], fields['height']) == ('336', '210')
        assert fields['nb_read_frames'] == '3'


def test_frame_size_must_not_change(tmp_path):
    movie = movie_writer.MovieWriter(tmp_path / 'test.mov', fps=5)
    movie.frame_shape = (10, 10, 4)
    movie._proc = object()  # pretend ffmpeg is already running
    with pytest.raises(ValueError):
        movie.write(np.zeros((12, 10, 4), dtype=np.uint8))
