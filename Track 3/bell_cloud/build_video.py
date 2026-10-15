"""Render a narrated, captioned submission video from local evidence.

Requires macOS say, ffmpeg, ffprobe and Pillow. No upload or network calls.
Scientific charts are generated separately by build_submission.py.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import textwrap

from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
SUBMISSION = HERE / "submission"
W, H = 1920, 1080
FONT_ROOT = Path("/System/Library/Fonts/Supplemental")


def font(size, bold=False):
    return ImageFont.truetype(str(FONT_ROOT / ("Arial Bold.ttf" if bold else "Arial.ttf")), size)


def run(args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def duration(path):
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)]).strip())


def wrap_pixels(text, face, width, draw):
    lines, current = [], ""
    for word in text.split():
        attempt = (current + " " + word).strip()
        if draw.textlength(attempt, font=face) > width and current:
            lines.append(current)
            current = word
        else:
            current = attempt
    return lines + ([current] if current else [])


def frame(scene, scene_index, caption, total_scenes):
    image = Image.new("RGB", (W, H), "#f6f5f1")
    draw = ImageDraw.Draw(image)
    draw.text((88, 43), "QOMPUTE 2026   /   TRACK 3", font=font(25, True), fill="#137f79")
    title_lines = wrap_pixels(scene["title"], font(62, True), 1744, draw)
    for index, line in enumerate(title_lines):
        draw.text((88, 94 + 72 * index), line, font=font(62, True), fill="#15273b")
    y = 105 + 72 * len(title_lines)
    draw.text((88, y), scene["subtitle"], font=font(32), fill="#536477")
    content_y = y + 66
    if scene.get("figure"):
        chart = Image.open(SUBMISSION / "figures" / scene["figure"]).convert("RGB")
        chart = ImageOps.contain(chart, (1744, 820 - content_y))
        image.paste(chart, ((W-chart.width)//2, content_y))
    else:
        row_y = content_y + 25
        for point in scene["points"]:
            lines = wrap_pixels(point, font(42), 1640, draw)
            for line in lines:
                draw.text((108, row_y), line, font=font(42), fill="#24384b")
                row_y += 54
            row_y += 22
        if row_y > 825:
            raise ValueError(f"Scene {scene_index+1} content would overlap captions.")
    draw.text((88, 850), scene["source"], font=font(22), fill="#566b7e")
    draw.text((1770, 850), f"{scene_index+1:02d}/{total_scenes:02d}", font=font(22), fill="#566b7e")
    # Captions are part of the video frames and also provided as a separate SRT.
    draw.rectangle((0, 900, W, H), fill="#132338")
    lines = wrap_pixels(caption, font(39), 1736, draw)
    if len(lines) > 3:
        raise ValueError("Caption exceeds three lines; split the sentence.")
    caption_y = 918 + (3-len(lines))*23
    for line in lines:
        draw.text(((W-draw.textlength(line,font=font(39))) / 2, caption_y), line, font=font(39), fill="white")
        caption_y += 47
    return image


def srt_time(seconds):
    ms = round(seconds*1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("/private/tmp/qompute-track3-video"))
    args = parser.parse_args()
    args.build_dir.mkdir(parents=True, exist_ok=True)
    scenes = json.loads((SUBMISSION / "video_script.json").read_text())
    concat_images, concat_audio, srt, timings, preview = [], [], [], [], []
    cursor, caption_index = 0.0, 0
    for scene_index, scene in enumerate(scenes):
        scene_start = cursor
        sentences = re.split(r"(?<=[.!?])\s+", scene["narration"].strip())
        for sentence_index, sentence in enumerate(sentences):
            # Keep captions short without inventing a different spoken script.
            chunks = textwrap.wrap(sentence, width=155, break_long_words=False, break_on_hyphens=False)
            for chunk_index, chunk in enumerate(chunks):
                content_hash = hashlib.sha256(("Samantha:148:"+chunk).encode()).hexdigest()[:10]
                stem = f"scene-{scene_index:02}-sentence-{sentence_index:02}-{chunk_index}-{content_hash}"
                text_path = args.build_dir / f"{stem}.txt"
                audio_path = args.build_dir / f"{stem}.aiff"
                text_path.write_text(chunk, encoding="utf-8")
                if not audio_path.exists() or duration(audio_path) <= .05:
                    run(["say", "-v", "Samantha", "-r", "148", "-f", str(text_path), "-o", str(audio_path)])
                seconds = duration(audio_path)
                if seconds <= .05:
                    raise RuntimeError("macOS speech synthesis returned empty audio. Allow the local speech service.")
                image_path = args.build_dir / f"{stem}.png"
                img = frame(scene, scene_index, chunk, len(scenes))
                img.save(image_path)
                if sentence_index == 0 and chunk_index == 0:
                    preview.append(img.resize((640, 360)))
                concat_images.extend([f"file '{image_path}'", f"duration {seconds:.6f}"])
                concat_audio.append(f"file '{audio_path}'")
                caption_index += 1
                srt.extend([str(caption_index), f"{srt_time(cursor)} --> {srt_time(cursor+seconds)}", chunk, ""])
                cursor += seconds
        timings.append({"scene": scene_index+1, "title": scene["title"], "start": scene_start, "end": cursor})
        print(f"Scene {scene_index+1}/{len(scenes)} rendered; total {cursor:.1f} seconds", flush=True)
    concat_images.append(concat_images[-2])
    image_list, audio_list = args.build_dir/"images.ffconcat", args.build_dir/"audio.ffconcat"
    image_list.write_text("\n".join(concat_images)+"\n")
    audio_list.write_text("\n".join(concat_audio)+"\n")
    audio = args.build_dir / "narration.wav"
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(audio_list), "-c:a", "pcm_s16le", str(audio)])
    video = SUBMISSION / "Track3_Presentation.mp4"
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(image_list),
         "-i", str(audio), "-vf", "fps=12,format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-shortest", str(video)])
    seconds = duration(video)
    if not 300 <= seconds <= 600:
        raise ValueError(f"Video duration {seconds:.1f}s is outside the required 5-10 minutes.")
    (SUBMISSION / "Track3_Presentation.srt").write_text("\n".join(srt), encoding="utf-8")
    sheet = Image.new("RGB", (1920, 360*((len(preview)+2)//3)), "#e5e7eb")
    for i, img in enumerate(preview): sheet.paste(img, ((i%3)*640, (i//3)*360))
    sheet.save(args.build_dir / "contact-sheet.jpg", quality=90)
    receipt = {"generated_utc": datetime.now(timezone.utc).isoformat(), "duration_seconds": seconds,
               "resolution": [W,H], "fps": 12, "voice": "macOS Samantha (synthetic)", "rate_words_per_minute": 148,
               "caption_segments": caption_index, "scenes": timings, "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
               "video_bytes": video.stat().st_size, "uploaded": False,
               "script_sha256": hashlib.sha256((SUBMISSION/"video_script.json").read_bytes()).hexdigest()}
    (SUBMISSION / "video-manifest.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({"video": str(video), "duration_seconds": seconds, "bytes": video.stat().st_size}))


if __name__ == "__main__":
    main()
