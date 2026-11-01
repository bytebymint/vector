"""Unit tests verifying Live Video Pipeline acquisition, buffering, and distribution."""
import queue
import threading
import time
import numpy as np
import pytest

from services.face_contracts import FramePacket, ProcessingMetrics
from services.video_pipeline import (
    VideoPipeline,
    RingBuffer,
    FrameConsumerQueue,
    extract_crop,
    extract_crop_coordinates,
)


def test_frame_packet_packaging_and_timestamping():
    """Verify FramePacket preserves native resolution, timestamps, and metadata."""
    pipeline = VideoPipeline(source_id="cctv_main", buffer_size=10, epoch="session_alpha")

    # Native 1080p frame
    img_1080p = np.zeros((1080, 1920, 3), dtype=np.uint8)
    img_1080p[100:200, 100:200] = 255  # Distinct marker

    t0 = time.time()
    packet = pipeline.publish_frame(img_1080p, timestamp=t0, pts=123.456)

    assert isinstance(packet, FramePacket)
    assert packet.sequence == 1
    assert packet.width == 1920
    assert packet.height == 1080
    assert packet.image.shape == (1080, 1920, 3)
    assert packet.timestamp == t0
    assert packet.pts == 123.456
    assert packet.source_id == "cctv_main"
    assert packet.epoch == "session_alpha"

    # Verify serialization contract
    d = packet.to_dict()
    assert d["sequence"] == 1
    assert d["width"] == 1920
    assert d["height"] == 1080
    assert d["source_id"] == "cctv_main"
    assert d["pts"] == 123.456

    # Native 4K frame preservation
    img_4k = np.zeros((2160, 3840, 3), dtype=np.uint8)
    packet_4k = pipeline.publish_frame(img_4k)
    assert packet_4k.sequence == 2
    assert packet_4k.width == 3840
    assert packet_4k.height == 2160
    assert packet_4k.image.shape == (2160, 3840, 3)

    # get_latest_frame should return the 4K uncompressed frame
    latest = pipeline.get_latest_frame()
    assert latest is not None
    assert latest.sequence == 2
    assert latest.width == 3840


def test_ring_buffer_bounds_and_drop_behavior():
    """Verify ring buffer strictly adheres to maxlen bound and drops oldest frames."""
    buffer = RingBuffer(maxlen=5)
    assert len(buffer) == 0
    assert buffer.capacity == 5
    assert buffer.dropped_count == 0

    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    # Append 8 packets
    for seq in range(1, 9):
        pkt = FramePacket(
            sequence=seq,
            timestamp=float(seq),
            image=dummy_img,
            source_id="cam",
            epoch="ep1",
            width=100,
            height=100,
        )
        buffer.append(pkt)

    # Capacity should not exceed 5
    assert len(buffer) == 5
    assert buffer.total_pushed == 8
    assert buffer.dropped_count == 3  # 3 frames dropped due to bound

    # Latest should be sequence 8
    latest = buffer.get_latest()
    assert latest is not None
    assert latest.sequence == 8

    # Sequences 1, 2, 3 should have been evicted
    assert buffer.get_by_sequence(1) is None
    assert buffer.get_by_sequence(2) is None
    assert buffer.get_by_sequence(3) is None

    # Sequences 4, 5, 6, 7, 8 should be present
    assert buffer.get_by_sequence(4) is not None
    assert buffer.get_by_sequence(8) is not None

    # Recent count retrieval
    recent = buffer.get_recent(3)
    assert len(recent) == 3
    assert [p.sequence for p in recent] == [6, 7, 8]

    # Clear
    buffer.clear()
    assert len(buffer) == 0
    assert buffer.get_latest() is None


def test_high_resolution_crop_extraction_with_margin_clamping():
    """Verify crop extraction with margin calculation, edge clamping, and preview scaling."""
    # Create 1920x1080 master frame
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Paint a 100x100 white square at (200, 200) to (300, 300)
    img[200:300, 200:300] = 255

    packet = FramePacket(
        sequence=1,
        timestamp=1.0,
        image=img,
        source_id="test",
        epoch="ep1",
        width=1920,
        height=1080,
    )

    # Case 1: Standard crop inside frame with 20% margin
    # Bbox: (200, 200, 300, 300) -> w=100, h=100, dx=20, dy=20
    # Expected: (180, 180, 320, 320) -> w=140, h=140
    crop = extract_crop(packet, bbox=(200, 200, 300, 300), margin=0.2)
    assert crop.shape == (140, 140, 3)

    # Verify coordinates helper
    x1, y1, x2, y2 = extract_crop_coordinates(packet, bbox=(200, 200, 300, 300), margin=0.2)
    assert (x1, y1, x2, y2) == (180, 180, 320, 320)

    # Case 2: Top-left boundary clamping
    # Bbox near corner: (10, 10, 50, 50) with margin 0.5 -> w=40, dx=20 -> -10 clamped to 0
    crop_tl = extract_crop(packet, bbox=(10, 10, 50, 50), margin=0.5)
    # x1=0, y1=0, x2=70, y2=70
    assert crop_tl.shape == (70, 70, 3)
    coords_tl = extract_crop_coordinates(packet, bbox=(10, 10, 50, 50), margin=0.5)
    assert coords_tl == (0, 0, 70, 70)

    # Case 3: Bottom-right boundary clamping
    # Bbox near bottom right: (1900, 1060, 1910, 1070) with margin 0.5 -> w=10, dx=5
    # x2 = 1915, y2 = 1075 -> inside. What if margin=2.0 -> w=10, dx=20 -> x2=1930 clamped to 1920
    crop_br = extract_crop(packet, bbox=(1900, 1060, 1910, 1070), margin=2.0)
    coords_br = extract_crop_coordinates(packet, bbox=(1900, 1060, 1910, 1070), margin=2.0)
    assert coords_br[2] == 1920
    assert coords_br[3] == 1080
    assert crop_br.shape[1] == 1920 - coords_br[0]
    assert crop_br.shape[0] == 1080 - coords_br[1]

    # Case 4: Preview scaling
    # Detection performed on 640x360 preview image
    # Preview bbox: (64, 36, 128, 72) -> 10% to 20% of preview
    # Original is 1920x1080 (3x scale)
    # Expected on master without margin: (192, 108, 384, 216) -> w=192, h=108
    crop_scaled = extract_crop(
        packet,
        bbox=(64, 36, 128, 72),
        margin=0.0,
        preview_shape=(360, 640),
    )
    assert crop_scaled.shape == (108, 192, 3)

    # Also test preview_size=(640, 360)
    crop_scaled_size = extract_crop(
        packet,
        bbox=(64, 36, 128, 72),
        margin=0.0,
        preview_size=(640, 360),
    )
    assert crop_scaled_size.shape == (108, 192, 3)

    # Case 5: Normalized coordinates [0.0, 1.0]
    crop_norm = extract_crop(packet, bbox=(0.1, 0.1, 0.2, 0.2), margin=0.0)
    assert crop_norm.shape == (108, 192, 3)

    # Case 6: Degenerate or out-of-bounds coordinates
    crop_oob = extract_crop(packet, bbox=(2500, 2500, 2600, 2600), margin=0.0)
    assert crop_oob.size == 0


def test_multiple_consumers_without_starvation():
    """Verify multiple downstream consumers receive frames without blocking acquisition or starving."""
    pipeline = VideoPipeline(source_id="multi_consumer_cam", buffer_size=10)

    # Register consumers with different buffer sizes
    fast_consumer = pipeline.register_consumer("fast_worker", max_size=20, drop_oldest=True)
    slow_consumer = pipeline.register_consumer("slow_detector", max_size=3, drop_oldest=True)
    no_drop_consumer = pipeline.register_consumer("strict_consumer", max_size=5, drop_oldest=False)

    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Publish 15 frames rapidly
    for seq in range(1, 16):
        pipeline.publish_frame(img, sequence=seq)

    # Verify slow consumer queue did not exceed max_size 3
    assert slow_consumer.qsize() == 3
    assert slow_consumer.dropped_count == 12  # 15 - 3 = 12 drops
    assert slow_consumer.total_received == 15

    # Since slow_consumer has drop_oldest=True, it holds the latest 3 frames (13, 14, 15)
    slow_packets = []
    while not slow_consumer.empty():
        slow_packets.append(slow_consumer.get_nowait().sequence)
    assert slow_packets == [13, 14, 15]

    # Verify fast consumer received all 15 frames
    assert fast_consumer.qsize() == 15
    assert fast_consumer.dropped_count == 0

    # Strict consumer with drop_oldest=False holds the FIRST 5 frames (1..5)
    assert no_drop_consumer.qsize() == 5
    assert no_drop_consumer.dropped_count == 10
    strict_packets = []
    while not no_drop_consumer.empty():
        strict_packets.append(no_drop_consumer.get_nowait().sequence)
    assert strict_packets == [1, 2, 3, 4, 5]


def test_consumer_queue_blocking_and_closing():
    """Verify FrameConsumerQueue blocking get, timeout, and clean shutdown."""
    q = FrameConsumerQueue("test_q", max_size=2)

    # Empty non-blocking get raises queue.Empty
    with pytest.raises(queue.Empty):
        q.get_nowait()

    # Empty blocking get with timeout returns None via get_optional
    assert q.get_optional(timeout=0.05) is None

    # Asynchronous put from background thread
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    pkt = FramePacket(sequence=1, timestamp=1.0, image=img, source_id="s", epoch="e", width=10, height=10)

    def _producer():
        time.sleep(0.05)
        q.put(pkt)

    t = threading.Thread(target=_producer)
    t.start()

    received = q.get(block=True, timeout=1.0)
    assert received.sequence == 1
    t.join()

    # Close queue
    q.close()
    assert q.is_closed is True
    # Putting to closed queue returns False
    assert q.put(pkt) is False


def test_pipeline_wait_for_frame_synchronization():
    """Verify wait_for_frame condition variable synchronization."""
    pipeline = VideoPipeline(source_id="sync_cam")
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    def _feeder():
        time.sleep(0.05)
        pipeline.publish_frame(img, sequence=10)

    t = threading.Thread(target=_feeder)
    t.start()

    # Wait for frame after 0
    packet = pipeline.wait_for_frame(after_sequence=0, timeout=1.0)
    assert packet is not None
    assert packet.sequence == 10
    t.join()

    # Timeout waiting for already-seen sequence
    timed_out = pipeline.wait_for_frame(after_sequence=10, timeout=0.05)
    assert timed_out is None


def test_pipeline_preview_generation():
    """Verify get_preview_frame generates downscaled copies without mutating master frame."""
    pipeline = VideoPipeline(source_id="preview_cam")
    master_img = np.full((1080, 1920, 3), 42, dtype=np.uint8)
    pipeline.publish_frame(master_img)

    # Target size preview
    preview = pipeline.get_preview_frame(target_size=(640, 360))
    assert preview is not None
    assert preview.shape == (360, 640, 3)

    # Target max dim preview
    preview_dim = pipeline.get_preview_frame(target_max_dim=960)
    assert preview_dim is not None
    assert max(preview_dim.shape[:2]) == 960

    # Verify master frame remains 1080p untouched
    latest = pipeline.get_latest_frame()
    assert latest.width == 1920
    assert latest.height == 1080
    assert latest.image.shape == (1080, 1920, 3)


def test_pipeline_metrics_and_reset():
    """Verify metrics calculation and pipeline reset behavior."""
    pipeline = VideoPipeline(source_id="metrics_cam", buffer_size=5, epoch="epoch_1")
    pipeline.register_consumer("det_queue", max_size=2)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(5):
        pipeline.publish_frame(img)

    metrics = pipeline.get_metrics()
    assert metrics["sequence"] == 5
    assert metrics["buffer_size"] == 5
    assert metrics["active_consumers"] == 1
    assert "det_queue" in metrics["consumer_stats"]

    proc_metrics = pipeline.get_processing_metrics()
    assert isinstance(proc_metrics, ProcessingMetrics)
    assert proc_metrics.frame_sequence == 5

    # Reset
    pipeline.reset(epoch="epoch_2")
    assert pipeline.epoch == "epoch_2"
    assert len(pipeline.ring_buffer) == 0
    assert pipeline.get_latest_frame() is None
    assert pipeline.get_consumer("det_queue").qsize() == 0


def test_pipeline_background_ingestion_and_context_manager():
    """Verify background ingestion from iterable and context manager teardown."""
    frames = [
        np.zeros((100, 100, 3), dtype=np.uint8),
        np.ones((100, 100, 3), dtype=np.uint8) * 100,
        np.ones((100, 100, 3), dtype=np.uint8) * 200,
    ]

    with VideoPipeline(source_id="bg_cam") as pipeline:
        consumer = pipeline.register_consumer("reader", max_size=5)
        pipeline.start_ingestion(frames, fps=100.0)

        # Allow ingestion thread to run
        time.sleep(0.1)
        pipeline.stop_ingestion()

        assert consumer.qsize() >= 3
        pkt = consumer.get_nowait()
        assert pkt.sequence == 1

    # Context manager exit closes pipeline and consumers
    assert consumer.is_closed is True
