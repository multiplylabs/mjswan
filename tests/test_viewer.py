"""Tests for mjswan.viewer — the ViewerConfig surface that reaches config.json."""

from __future__ import annotations

from mjswan.viewer import ViewerConfig


class TestHandTracking:
    def test_defaults_on(self):
        assert ViewerConfig().hand_tracking is True

    def test_serialized_for_the_client(self):
        """The browser reads this key to decide whether to inject the XR hand rig."""
        assert ViewerConfig().to_dict()["handTracking"] is True
        assert ViewerConfig(hand_tracking=False).to_dict()["handTracking"] is False
