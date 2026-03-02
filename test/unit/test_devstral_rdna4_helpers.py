import json
import tempfile
import unittest

from devstral_rdna4 import _build_contiguous_layer_map, _is_fp8_export


class TestDevstralRDNA4Helpers(unittest.TestCase):
  def test_build_contiguous_layer_map_even_split(self):
    devices = ["AMD:0", "AMD:1", "AMD:2", "AMD:3"]
    mapping = _build_contiguous_layer_map(40, devices)
    self.assertEqual(len(mapping), 40)
    self.assertEqual(mapping.count("AMD:0"), 10)
    self.assertEqual(mapping.count("AMD:1"), 10)
    self.assertEqual(mapping.count("AMD:2"), 10)
    self.assertEqual(mapping.count("AMD:3"), 10)
    # contiguous partitioning
    self.assertEqual(mapping[:10], ["AMD:0"] * 10)
    self.assertEqual(mapping[10:20], ["AMD:1"] * 10)
    self.assertEqual(mapping[20:30], ["AMD:2"] * 10)
    self.assertEqual(mapping[30:], ["AMD:3"] * 10)

  def test_build_contiguous_layer_map_single_device(self):
    mapping = _build_contiguous_layer_map(7, ["AMD:0"])
    self.assertEqual(mapping, ["AMD:0"] * 7)

  def test_is_fp8_export_true_from_index(self):
    with tempfile.TemporaryDirectory() as td:
      with open(f"{td}/model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump({
          "weight_map": {
            "language_model.model.layers.0.self_attn.q_proj.weight": "model-00001-of-00001.safetensors",
            "language_model.model.layers.0.self_attn.q_proj.activation_scale": "model-00001-of-00001.safetensors",
          }
        }, f)
      self.assertTrue(_is_fp8_export(td))

  def test_is_fp8_export_false_from_index(self):
    with tempfile.TemporaryDirectory() as td:
      with open(f"{td}/model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump({
          "weight_map": {
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00001.safetensors",
          }
        }, f)
      self.assertFalse(_is_fp8_export(td))


if __name__ == '__main__':
  unittest.main()
