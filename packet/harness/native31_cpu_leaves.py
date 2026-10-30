"""Declared TP1 group leaf for legacy CPU fake tensor/model fixtures only."""
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch
import lane_comm


def tp1_group(test):
    original = lane_comm.canonical_selection
    context = patch.object(lane_comm, 'canonical_selection',
                           partial(original, group=SimpleNamespace(world_size=1)))
    context.start()
    test.addCleanup(context.stop)
