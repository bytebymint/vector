import asyncio
import argparse
from pathlib import Path
import time
from urllib.parse import urljoin

import aiohttp
import cv2
from aiortc import (
    RTCConfiguration,
    RTCPeerConnection,
    RTCSessionDescription,
)
from services.stream_manager import enable_rsa_dtls_ciphers, negotiate
from video_enhancement import VideoEnhancer

BASE_URL = "https://livestream.pattaya.go.th/"
STREAM_ID = "4f2e6981-eefd-4eb0-ae5c-ccdd5cb3344a"
WINDOW_NAME = "Pattaya Live CCTV - RC-293 | Super Resolution"


def comparison_frame(original, enhanced):
    half = enhanced.shape[1] // 2
    result = enhanced.copy()
    result[:, :half] = original[:, :half]
    cv2.line(result, (half, 0), (half, result.shape[0]), (210, 210, 210), 2)
    return result


def label_frame(image, mode):
    image = image.copy()
    height, width = image.shape[:2]
    ui_scale = max(1, min(2.5, width / 1536))
    cv2.rectangle(image, (0, height - round(32 * ui_scale)), (width, height), (22, 22, 22), -1)
    cv2.putText(
        image, f"{mode}  |  E: original   C: compare   P: quality/speed   F: fullscreen   Q: quit",
        (round(12 * ui_scale), height - round(10 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX,
        0.5 * ui_scale, (235, 235, 235), max(1, round(ui_scale)),
        cv2.LINE_AA,
    )
    return image


async def main():
    parser = argparse.ArgumentParser(description="Live Pattaya WebRTC viewer with digital enhancement")
    parser.add_argument("target", nargs="?", default=STREAM_ID, help="Fresh playUrl or camera UUID")
    parser.add_argument("--scale", type=int, choices=(1, 2, 3, 4, 5), default=5,
                        help="Output multiplier; 5 produces 3840x2160 from this camera")
    parser.add_argument("--sr-model", choices=("fsrcnn", "espcn", "lanczos"), default="espcn")
    parser.add_argument("--model-scale", type=int, choices=(2, 4), default=4)
    parser.add_argument("--compute", choices=("opencl", "cpu"), default="opencl")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--preview-file", type=Path, help="Save one original/enhanced comparison frame")
    args = parser.parse_args()
    enable_rsa_dtls_ciphers()
    target = args.target
    details = {}
    cv2.setNumThreads(args.threads)
    enhancer = VideoEnhancer(args.scale, args.sr_model, args.model_scale, args.compute)

    async with aiohttp.ClientSession() as session:
        if target.startswith(("https://", "http://", "/media/")):
            # A fresh playUrl can be passed directly when the browser session
            # is required to mint it.
            url = urljoin(BASE_URL, target)
        else:
            # Fetch a token automatically when the public endpoint permits it.
            public_stream_url = urljoin(BASE_URL, f"api/public-stream/{target}")
            async with session.get(public_stream_url) as response:
                if response.status == 403:
                    raise RuntimeError(
                        "Pattaya requires the browser session to mint this URL. "
                        "Pass the latest playUrl from the open camera page."
                    )
                response.raise_for_status()
                stream_info = await response.json(content_type=None)
            details = stream_info.get("details", {})
            play_url = details.get("playUrl")
            if not play_url:
                raise RuntimeError(
                    f"No playUrl in public-stream response: {stream_info}"
                )
            url = urljoin(BASE_URL, play_url)

        pc = RTCPeerConnection(
            RTCConfiguration(
                iceServers=[]
            )
        )
        finished = asyncio.Event()
        pc.addTransceiver("video", direction="recvonly")
        pc.addTransceiver("audio", direction="recvonly")

        @pc.on("iceconnectionstatechange")
        async def ice_state_changed():
            print("ICE:", pc.iceConnectionState, flush=True)

        @pc.on("connectionstatechange")
        async def state_changed():
            print("Connection:", pc.connectionState)
            if pc.connectionState in ("failed", "closed"):
                finished.set()

        @pc.on("track")
        def received_track(track):
            if track.kind != "video":
                return

            latest_frame = asyncio.Queue(maxsize=1)
            dropped_frames = 0

            async def receive_frames():
                nonlocal dropped_frames
                try:
                    while not finished.is_set():
                        frame = await track.recv()
                        if latest_frame.full():
                            latest_frame.get_nowait()
                            dropped_frames += 1
                        latest_frame.put_nowait(frame)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if not finished.is_set():
                        print("Video source stopped:", exc)
                    finished.set()

            async def display():
                reader = asyncio.create_task(receive_frames())
                try:
                    frame_count = 0
                    enhanced_enabled = True
                    compare_enabled = False
                    fullscreen = False
                    processing_times = []
                    display_started = time.perf_counter()
                    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(WINDOW_NAME, 1280, 720)
                    while not finished.is_set():
                        try:
                            frame = await asyncio.wait_for(latest_frame.get(), timeout=2)
                        except asyncio.TimeoutError:
                            continue
                        image = frame.to_ndarray(format="bgr24")
                        # `image` remains the untouched BGR frame for YOLO.
                        started = time.perf_counter()
                        enhanced = await asyncio.to_thread(enhancer.process, image)
                        processing_times.append(time.perf_counter() - started)
                        frame_count += 1
                        if frame_count == 1:
                            print(f"First video frame: {image.shape[1]}x{image.shape[0]}", flush=True)
                            print(f"Enhanced display: {enhanced.shape[1]}x{enhanced.shape[0]}", flush=True)
                            print(f"Restoration: {enhancer.description}; latest-frame buffer enabled", flush=True)
                            if args.preview_file:
                                args.preview_file.parent.mkdir(parents=True, exist_ok=True)
                                preview = label_frame(
                                    comparison_frame(enhancer.resize(image), enhanced),
                                    "ORIGINAL (left) / ENHANCED (right)",
                                )
                                if not cv2.imwrite(str(args.preview_file), preview):
                                    raise RuntimeError("Could not save comparison preview")
                                cv2.imwrite(str(args.preview_file.with_name(
                                    args.preview_file.stem + "-source.png"
                                )), image)
                        if frame_count in (10, 60):
                            average_ms = 1000 * sum(processing_times) / len(processing_times)
                            fps = frame_count / (time.perf_counter() - display_started)
                            print(f"Enhancement: {average_ms:.1f} ms/frame; display {fps:.1f} fps; skipped stale frames {dropped_frames}", flush=True)
                        elif frame_count > 60:
                            processing_times.clear()
                        if compare_enabled:
                            output = comparison_frame(enhancer.resize(image), enhanced)
                            mode = "ORIGINAL (left) / ENHANCED (right)"
                        elif enhanced_enabled:
                            output, mode = enhanced, f"{enhanced.shape[1]}x{enhanced.shape[0]} / SR x{enhancer.model_scale}"
                        else:
                            output, mode = enhancer.resize(image), "ORIGINAL"
                        cv2.imshow(WINDOW_NAME, label_frame(output, mode))
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q") or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                            finished.set()
                            break
                        if key == ord("e"):
                            enhanced_enabled = not enhanced_enabled
                            compare_enabled = False
                        elif key == ord("c"):
                            compare_enabled = not compare_enabled
                        elif key == ord("f"):
                            fullscreen = not fullscreen
                            cv2.setWindowProperty(
                                WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                            )
                        elif key == ord("p"):
                            enhancer.toggle_quality()
                            print("Restoration:", enhancer.description, flush=True)
                except Exception as exc:
                    print("Video stopped:", exc)
                    finished.set()
                finally:
                    reader.cancel()
                    await asyncio.gather(reader, return_exceptions=True)

            asyncio.create_task(display())

        try:
            await negotiate(pc, session, url)
            print("Waiting for video. Press Q to exit.")
            await finished.wait()
        finally:
            await pc.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())


