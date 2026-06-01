import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.preference_memory import PreferenceMemory


class PreferenceMemoryTest(unittest.TestCase):
    def setUp(self):
        self.podcast_latents = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.podcast_ids = np.array(["podcast_a", "podcast_b", "podcast_c"])
        self.user_id_map = {"user_1": 0, "user_2": 1}

    def _load_memory(self, rows):
        memory = PreferenceMemory(
            podcast_latents=self.podcast_latents,
            podcast_ids=self.podcast_ids,
            user_id_map=self.user_id_map,
            alpha=0.5,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "events.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            memory.load_events(path)
        return memory

    def test_positive_edit_moves_query_toward_podcast(self):
        memory = self._load_memory(
            [{"user_id": "user_1", "podcast_id": "podcast_b", "event_type": "like"}]
        )
        base = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        conditioned = memory.condition_queries(torch.tensor([0]), base)

        self.assertGreater(conditioned[0, 1].item(), 0.0)
        self.assertAlmostEqual(torch.linalg.norm(conditioned[0]).item(), 1.0, places=5)

    def test_negative_edit_moves_query_away_from_podcast(self):
        memory = self._load_memory(
            [{"user_id": "user_1", "podcast_id": "podcast_b", "event_type": "skip"}]
        )
        base = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        conditioned = memory.condition_queries(torch.tensor([0]), base)

        self.assertLess(conditioned[0, 1].item(), 0.0)

    def test_remove_event_excludes_exact_podcast(self):
        memory = self._load_memory(
            [{"user_id": "user_1", "podcast_id": "podcast_b", "event_type": "remove"}]
        )

        podcast_ids, scores = memory.filter_results(
            user_idx=0,
            podcast_ids=["podcast_b", "podcast_a", "podcast_c"],
            scores=[0.9, 0.8, 0.7],
            top_k=2,
        )

        self.assertEqual(podcast_ids, ["podcast_a", "podcast_c"])
        self.assertEqual(scores, [0.8, 0.7])


if __name__ == "__main__":
    unittest.main()
