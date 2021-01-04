import os
import cv2
import glob
import argparse
from shutil import rmtree
from SyncNetInstance import SyncNetInstance

# ==================== PARSE ARGUMENT ====================

parser = argparse.ArgumentParser(description = "SyncNet")
parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='')
parser.add_argument('--batch_size', type=int, default='20', help='')
parser.add_argument('--vshift', type=int, default='15', help='')
parser.add_argument('--data_dir', type=str, default='data/work', help='')
opt = parser.parse_args()
setattr(opt, 'avi_dir',  os.path.join(opt.data_dir, 'pyavi'))
setattr(opt, 'tmp_dir',  os.path.join(opt.data_dir, 'pytmp'))
setattr(opt, 'work_dir', os.path.join(opt.data_dir, 'pywork'))
setattr(opt, 'crop_dir', os.path.join(opt.data_dir, 'pycrop'))

s = SyncNetInstance()
s.loadParameters(opt.initial_model)


def get_offset(video_path, video_name):
    root = os.path.abspath(os.path.dirname(__file__))
    cmd = f"cd {root} && python3 run_pipeline.py --videofile '{video_path}' --reference '{video_name}' > /dev/null 2>&1"
    os.system(cmd)

    setattr(opt, 'videofile', video_path)
    setattr(opt, 'reference', video_name)

    flist = glob.glob(os.path.join(opt.crop_dir, opt.reference, '0*.avi'))
    flist.sort()
    assert len(flist) == 1

    offset, conf, dist = s.evaluate(opt, videofile=flist[0])
    rmtree(opt.data_dir)
    return offset


def remerge_media(output_path, video_path, av_offset, target_fps: float = 30.0, keep_parts = False):
    out_name = os.path.splitext(os.path.basename(output_path))[0]
    out_dir = os.path.dirname(video_path)

    vpath = os.path.join(out_dir, f"{out_name}_partv.mp4")
    tpath = os.path.join(out_dir, f"{out_name}_parta_tmp.wav")
    apath = os.path.join(out_dir, f"{out_name}_parta.wav")
    if av_offset == 0:
        tpath = apath

    # check fps of video
    reader = cv2.VideoCapture(video_path)
    fps = reader.get(cv2.CAP_PROP_FPS)
    # part video
    if fps == target_fps:
        os.system(f"ffmpeg -y -loglevel error -i '{video_path}' -map 0:v -c:v copy               '{vpath}'")
    else:
        os.system(f"ffmpeg -y -loglevel error -i '{video_path}' -map 0:v -r {target_fps} -crf 18 '{vpath}'")
    # part audio
    os.system(f"ffmpeg -y -loglevel error -i '{video_path}' -map 0:a -c:a pcm_s16le -ac 1 '{tpath}'")
    # shift the audio
    if av_offset > 0:
        os.system(f"ffmpeg -y -loglevel error -i '{tpath}' -c:a pcm_s16le -af 'adelay={av_offset * 40}'    '{apath}'")
    elif av_offset < 0:
        os.system(f"ffmpeg -y -loglevel error -i '{tpath}' -c:a pcm_s16le -af 'atrim={av_offset * -0.040}' '{apath}'")
    else:
        pass
    # merge
    os.system(f"ffmpeg -y -loglevel error -i '{apath}' -i '{vpath}' -c:v copy -shortest '{output_path}'")
    # remove
    if apath != tpath:
        os.remove(tpath)
    if not keep_parts:
        os.remove(vpath)
        os.remove(apath)


def do_work(video_path, target_fps: float = 30.0, check: bool = True, failure_fp = None):
    video_path = os.path.abspath(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print("Process '{}'".format(video_path))

    # -------------------------------------------------- get offset -------------------------------------------------- #
    # remerge with target fps
    tmp_path = os.path.join(os.path.dirname(video_path), f'.{video_name}.mp4')
    remerge_media(tmp_path, video_path, 0, target_fps=target_fps, keep_parts=False)
    # get the offset of remerged
    offset = get_offset(tmp_path, video_name)
    os.remove(tmp_path)
    print("Detect AV offset {} for '{}'".format(offset, video_path))

    # -------------------------------------------------- shift media ------------------------------------------------- #
    video_dir = os.path.dirname(video_path)
    new_path = os.path.join(video_dir, f"{video_name}_aligned.mp4")
    remerge_media(new_path, video_path, offset, target_fps=target_fps, keep_parts=False)

    # debug
    if check:
        offset = get_offset(new_path, video_name + "_aligned")
        if offset != 0:
            print("[Failure]: The offset of aligned video is {}".format(offset))
            if failure_fp is not None:
                failure_fp.write(video_path + "\n")


video_list = [
    'videos/example.avi',
    'videos/obama.mp4'
]
with open("failure.txt", "w") as failure_fp:
    for i, video_path in enumerate(video_list):
        print('[{}/{}]'.format(i+1, len(video_list)), end=' ')
        do_work(video_path, check=True, failure_fp=failure_fp)
