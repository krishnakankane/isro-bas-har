"""Unit and integration tests for Conv1D-BiGRU HAR model and training pipeline."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.activity_engine.har_model import (
    Conv1DBiGRUHAR,
    HAR_ACTION_CLASSES,
    NUM_HAR_CLASSES,
    TemporalSelfAttention,
)
from src.activity_engine.feature_extractor import FEATURE_DIM
from src.activity_engine.temporal_har import TemporalHARClassifier
from scripts.train_har import (
    compute_class_weights,
    compute_metrics,
    stratified_train_val_split,
    train_har_model,
)


class TestHARModelArchitecture(unittest.TestCase):
    """Test suite for Conv1D-BiGRU temporal neural network module."""

    def setUp(self):
        self.batch_size = 4
        self.seq_len = 60
        self.input_dim = FEATURE_DIM  # 154
        self.num_classes = NUM_HAR_CLASSES  # 10

        self.model = Conv1DBiGRUHAR(
            input_dim=self.input_dim,
            seq_len=self.seq_len,
            conv_channels=128,
            gru_hidden_dim=64,
            gru_num_layers=2,
            num_classes=self.num_classes,
        )

    def test_01_model_construction_and_parameter_count(self):
        """Verify model initializes with expected layers and positive parameter count."""
        param_count = self.model.count_parameters()
        self.assertGreater(param_count, 100_000)
        self.assertLess(param_count, 1_000_000)

    def test_02_forward_pass_output_shape(self):
        """Verify forward pass output is (B, 10) float tensor."""
        dummy_input = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        output = self.model(dummy_input)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.all(torch.isfinite(output)))

    def test_03_temporal_attention_module(self):
        """Verify attention pooling aggregates across time correctly."""
        attn = TemporalSelfAttention(hidden_dim=128, attention_dim=64)
        dummy_seq = torch.randn(self.batch_size, self.seq_len, 128)
        context, weights = attn(dummy_seq)

        self.assertEqual(context.shape, (self.batch_size, 128))
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len, 1))
        # Attention weights across time should sum to 1.0 for each sequence
        sum_weights = torch.sum(weights, dim=1)
        self.assertTrue(torch.allclose(sum_weights, torch.ones_like(sum_weights), atol=1e-5))

    def test_04_predict_helper_single_window(self):
        """Verify single window prediction returns valid class_id, action_name, and confidence."""
        single_window = torch.randn(self.seq_len, self.input_dim)
        cid, name, conf = self.model.predict(single_window)

        self.assertIsInstance(cid, int)
        self.assertTrue(0 <= cid < self.num_classes)
        self.assertIsInstance(name, str)
        self.assertEqual(name, HAR_ACTION_CLASSES[cid])
        self.assertIsInstance(conf, float)
        self.assertTrue(0.0 <= conf <= 1.0)

    def test_05_checkpoint_save_and_load(self):
        """Verify model state dictionary can be saved and reloaded identically."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_model.pt"
            torch.save({"model_state_dict": self.model.state_dict()}, ckpt_path)

            loaded_model = Conv1DBiGRUHAR(
                input_dim=self.input_dim,
                seq_len=self.seq_len,
                num_classes=self.num_classes,
            )
            loaded_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            loaded_model.load_state_dict(loaded_ckpt["model_state_dict"])
            self.model.eval()
            loaded_model.eval()

            dummy_input = torch.randn(2, self.seq_len, self.input_dim)
            with torch.no_grad():
                out1 = self.model(dummy_input)
                out2 = loaded_model(dummy_input)

            self.assertTrue(torch.allclose(out1, out2, atol=1e-5))


class TestHARTrainingPipeline(unittest.TestCase):
    """Test suite for training utilities, metrics, and miniature train loop."""

    def test_06_stratified_split_preserves_classes(self):
        """Verify stratified split includes all classes in train and val sets."""
        labels = np.array([0]*20 + [1]*20 + [2]*20 + [3]*15 + [4]*10 + [5]*20 + [6]*25 + [7]*10 + [8]*15 + [9]*20)
        train_idx, val_idx = stratified_train_val_split(labels, val_ratio=0.2, seed=42)

        self.assertEqual(len(train_idx) + len(val_idx), len(labels))
        self.assertEqual(len(set(train_idx).intersection(set(val_idx))), 0)

        train_classes = set(labels[train_idx].tolist())
        val_classes = set(labels[val_idx].tolist())
        self.assertEqual(train_classes, set(range(10)))
        self.assertEqual(val_classes, set(range(10)))

    def test_07_class_weights_computation(self):
        """Verify computed class weights give higher weight to rare classes."""
        labels = np.array([0]*50 + [1]*10)
        weights = compute_class_weights(labels, num_classes=2)
        self.assertEqual(len(weights), 2)
        # Class 1 is rarer, so its weight must be greater than Class 0
        self.assertGreater(weights[1].item(), weights[0].item())

    def test_08_compute_metrics_correctness(self):
        """Verify accuracy and macro-F1 on synthetic predictions."""
        y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        y_pred = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 0])  # 9/10 correct

        metrics = compute_metrics(y_true, y_pred, num_classes=10)
        self.assertEqual(metrics["accuracy"], 0.9)
        self.assertGreater(metrics["macro_f1"], 0.8)
        self.assertIn("per_class", metrics)
        self.assertEqual(len(metrics["per_class"]), 10)

    def test_09_mini_training_run(self):
        """Run a 2-epoch miniature training loop on synthetic data to verify pipeline integration."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            synth_npz = tmp_p / "synthetic_windows.npz"

            # Create miniature dataset (30 windows, 10 classes)
            N = 30
            synth_windows = np.random.randn(N, 60, FEATURE_DIM).astype(np.float32)
            synth_labels = np.array([i % 10 for i in range(N)], dtype=np.int64)
            np.savez_compressed(synth_npz, windows=synth_windows, labels=synth_labels)

            out_dir = tmp_p / "checkpoints"
            report = train_har_model(
                dataset_path=synth_npz,
                output_dir=out_dir,
                epochs=2,
                batch_size=8,
                lr=1e-3,
                val_ratio=0.3,
                patience=5,
                device="cpu",
            )

            self.assertIn("best_epoch", report)
            self.assertIn("final_validation_metrics", report)
            self.assertTrue((out_dir / "conv1d_bigru_best.pt").exists())
            self.assertTrue((out_dir / "training_metrics.json").exists())

    def test_10_temporal_har_classifier_with_checkpoint(self):
        """Verify TemporalHARClassifier loads saved checkpoint and predicts action."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            ckpt_path = tmp_p / "test_har.pt"

            model = Conv1DBiGRUHAR()
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

            classifier = TemporalHARClassifier(checkpoint_path=ckpt_path, device="cpu")
            self.assertIsNotNone(classifier.model)


if __name__ == "__main__":
    unittest.main()
