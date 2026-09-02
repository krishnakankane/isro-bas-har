"""Unit and integration tests for dataset collection, frame extraction, and annotation tooling.

Covers:
- Session video & metadata recording utility (headless synthetic mode)
- Intelligent frame extractor & contact sheet builder
- Object dataset staging & anti-leakage distribution
- Temporal action interval annotator
- Protocol sequence anomaly annotator
- Master dataset integrity & anti-leakage validator
- Video-to-feature sliding-window tensor extractor (60 × 154)
- Dataset inventory statistics reporter
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.record_session import record_session
from scripts.extract_frames import extract_frames
from scripts.prepare_object_dataset import prepare_staging_directory, distribute_by_actor
from scripts.annotate_temporal import run_temporal_annotator, ACTION_TAXONOMY
from scripts.annotate_sequence_eval import create_sequence_manifest
from scripts.validate_dataset import DatasetValidator
from scripts.extract_har_features import process_video_to_features
from scripts.dataset_statistics import compute_dataset_statistics
from src.activity_engine.feature_extractor import FEATURE_DIM


class TestDatasetTooling(unittest.TestCase):
    """Test suite for all offline dataset collection and annotation tools."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="isro_test_data_"))
        self.data_dir = self.temp_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_record_session_synthetic(self):
        """Test recording utility in headless synthetic mode."""
        meta = record_session(
            source="synthetic",
            actor_id="TEST_ACTOR_01",
            session_id="SES_TEST_001",
            camera_id="TEST_CAM",
            lighting="TEST_LIGHT",
            handedness="RIGHT",
            protocol_id="HOME_DEMO_PROTOCOL_01",
            notes="Unit test recording",
            width=640,
            height=480,
            target_fps=30,
            output_dir=self.data_dir / "raw",
            show_window=False,
            auto_record_frames=45,
        )

        self.assertEqual(meta["session_id"], "SES_TEST_001")
        self.assertEqual(meta["actor_id"], "TEST_ACTOR_01")
        self.assertGreaterEqual(meta["recorded_frames"], 40)
        self.assertTrue(Path(meta["video_path"]).exists())

        # Verify video can be opened with OpenCV
        cap = cv2.VideoCapture(meta["video_path"])
        self.assertTrue(cap.isOpened())
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.assertGreaterEqual(frame_count, 40)
        cap.release()

    def test_02_extract_frames_and_contact_sheet(self):
        """Test intelligent frame extraction and contact sheet generation."""
        # 1. First create a synthetic video
        meta = record_session(
            source="synthetic",
            actor_id="ACTOR_01",
            session_id="SES_EXTRACT_TEST",
            output_dir=self.data_dir / "raw",
            show_window=False,
            auto_record_frames=60,
            width=320,
            height=240,
        )
        video_path = Path(meta["video_path"])

        # 2. Extract frames
        staging_dir = self.data_dir / "staging" / "SES_EXTRACT_TEST"
        summary = extract_frames(
            video_path=video_path,
            output_dir=staging_dir,
            interval_frames=10,
            min_diff_threshold=0.0,  # Synthetic frames are identical, accept diff 0
            generate_contact_sheet=True,
        )

        self.assertGreaterEqual(summary["extracted_frames_count"], 5)
        extracted_files = list(staging_dir.glob("*.jpg"))
        self.assertGreaterEqual(len(extracted_files), 5)
        self.assertIsNotNone(summary["contact_sheet_path"])
        self.assertTrue(Path(summary["contact_sheet_path"]).exists())

    def test_03_prepare_object_dataset_staging(self):
        """Test staging images to train/val/test splits and template generation."""
        staging_dir = self.data_dir / "staging" / "SES_01"
        staging_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy image
        dummy_img = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.imwrite(str(staging_dir / "frame_001.jpg"), dummy_img)
        cv2.imwrite(str(staging_dir / "frame_002.jpg"), dummy_img)

        obj_base = self.data_dir / "object_detection"
        summary = prepare_staging_directory(
            staging_dir=staging_dir,
            target_split="train",
            generate_templates=True,
            base_data_dir=obj_base,
        )

        self.assertEqual(summary["images_copied"], 2)
        self.assertEqual(summary["label_templates_created"], 2)

        # Verify .txt template was created
        lbl_file = obj_base / "labels" / "train" / "frame_001.txt"
        self.assertTrue(lbl_file.exists())

    def test_04_annotate_temporal_manifest(self):
        """Test temporal action interval annotator manifest creation."""
        # Create video
        meta = record_session(
            source="synthetic",
            actor_id="ACTOR_02",
            session_id="SES_ANNO_TEST",
            output_dir=self.data_dir / "raw",
            show_window=False,
            auto_record_frames=60,
            width=320,
            height=240,
        )

        mock_segs = [
            {"segment_id": 1, "action_id": 0, "action_name": "ACTION_IDLE", "start_frame": 0, "end_frame": 25},
            {"segment_id": 2, "action_id": 1, "action_name": "ACTION_SANITIZE", "start_frame": 30, "end_frame": 55},
        ]

        manifest = run_temporal_annotator(
            video_path=Path(meta["video_path"]),
            actor_id="ACTOR_02",
            output_dir=self.data_dir / "temporal_har" / "manifests",
            show_window=False,
            auto_save_and_exit=True,
            mock_segments=mock_segs,
        )

        self.assertEqual(manifest["segments_count"], 2)
        self.assertEqual(manifest["segments"][0]["action_name"], "ACTION_IDLE")
        self.assertEqual(manifest["segments"][1]["action_name"], "ACTION_SANITIZE")

    def test_05_sequence_anomaly_generator(self):
        """Test sequence validation ground-truth manifest generator."""
        out_dir = self.data_dir / "sequence_validation" / "manifests"

        # Nominal trial
        m1 = create_sequence_manifest(
            trial_id="TEST_NOMINAL",
            anomaly_type="CORRECT_SEQUENCE",
            output_dir=out_dir,
        )
        self.assertEqual(m1["expected_fsm_outcome"], "VALID_COMPLETED")
        self.assertEqual(len(m1["observed_sequence"]), 10)

        # Skipped step trial
        m2 = create_sequence_manifest(
            trial_id="TEST_SKIP_BOX",
            anomaly_type="SKIPPED_STEP",
            faulty_step_id=5,
            skip_step_ids=[5],
            output_dir=out_dir,
        )
        self.assertEqual(m2["expected_fsm_outcome"], "SKIPPED_STEP")
        self.assertEqual(len(m2["observed_sequence"]), 9)

    def test_06_dataset_validator(self):
        """Test dataset validator detecting errors and validating clean structure."""
        validator = DatasetValidator(data_root=self.data_dir, strict=False)
        passed = validator.validate_all()
        # Fresh clean structure should pass without fatal errors
        self.assertTrue(passed)

        # Invalidate YOLO label and assert detection
        bad_lbl = self.data_dir / "object_detection" / "labels" / "train" / "bad.txt"
        bad_img = self.data_dir / "object_detection" / "images" / "train" / "bad.jpg"
        bad_img.write_text("fake image", encoding="utf-8")
        bad_lbl.write_text("99 0.5 0.5 0.2 0.2\n", encoding="utf-8")  # Class 99 is illegal (0..6 allowed)

        validator2 = DatasetValidator(data_root=self.data_dir, strict=False)
        passed2 = validator2.validate_all()
        self.assertFalse(passed2)
        self.assertTrue(any("Invalid class ID 99" in e for e in validator2.errors))

    def test_07_extract_har_features_tensors(self):
        """Test video-to-feature sliding-window tensor extraction."""
        # 1. Create short synthetic video
        meta = record_session(
            source="synthetic",
            actor_id="ACTOR_01",
            session_id="SES_FEAT_TEST",
            output_dir=self.data_dir / "raw",
            show_window=False,
            auto_record_frames=75,
            width=320,
            height=240,
        )
        video_path = Path(meta["video_path"])

        # 2. Extract features
        feat_out = self.data_dir / "temporal_har" / "features"
        summary = process_video_to_features(
            video_path=video_path,
            output_dir=feat_out,
            window_size=60,
            stride=10,
        )

        self.assertGreaterEqual(summary["windows_generated"], 1)
        self.assertEqual(summary["tensor_shape"][1], 60)
        self.assertEqual(summary["tensor_shape"][2], FEATURE_DIM)  # 154 dimensions

        # Verify saved NPZ file
        npz_path = Path(summary["output_npz_path"])
        self.assertTrue(npz_path.exists())
        with np.load(npz_path) as data:
            windows = data["windows"]
            self.assertEqual(windows.shape, (summary["windows_generated"], 60, 154))
            self.assertEqual(windows.dtype, np.float32)

    def test_08_dataset_statistics_reporter(self):
        """Test statistics report generator."""
        stats = compute_dataset_statistics(data_root=self.data_dir)
        self.assertIn("raw_videos_count", stats)
        self.assertIn("object_images_per_split", stats)
        self.assertIn("temporal_action_segments", stats)

    def test_09_real_features_integrity_if_available(self):
        """Verify real feature dataset tensor integrity if the artifact exists."""
        project_root = Path(__file__).resolve().parent.parent
        real_npz = project_root / "data" / "temporal_har" / "features" / "SES_001_NOMINAL_windows.npz"
        if not real_npz.exists():
            self.skipTest("Real dataset NPZ not found.")

        with np.load(real_npz) as data:
            windows = data["windows"]
            labels = data["labels"]
            ranges = data["ranges"]
            self.assertEqual(windows.ndim, 3)
            self.assertEqual(windows.shape[1], 60)
            self.assertEqual(windows.shape[2], 154)
            self.assertEqual(windows.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(windows)))
            self.assertEqual(len(windows), len(labels))
            self.assertEqual(len(windows), len(ranges))
            # Verify all 10 action classes are present
            unique_classes = set(labels.tolist())
            self.assertEqual(unique_classes, set(range(10)))


if __name__ == "__main__":
    unittest.main()
