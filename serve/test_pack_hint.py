"""The advisory protocol must not write for default/vision/idle requests or consume a response."""
import threading
import unittest
from unittest.mock import Mock

from serve.server import EngineDied, StrataEngine


class PackHint(unittest.TestCase):
    def engine(self, advertised=3, busy=True):
        engine = StrataEngine.__new__(StrataEngine)
        engine.info = {"pack_hints": str(advertised)}
        engine.ctl = threading.Lock()
        if busy:
            engine.ctl.acquire()
        engine._send = Mock()
        return engine

    def test_waiting_text_sends_exactly_one_advisory_line(self):
        engine = self.engine()
        engine._pack_hint([1, 42, 7], None)
        engine._send.assert_called_once_with("BHINT 1,42,7")
        self.assertTrue(engine.ctl.locked())

    def test_default_vision_and_idle_requests_send_nothing(self):
        for advertised, busy, embeddings in [(0, True, None), (3, False, None), (3, True, "image.bin")]:
            with self.subTest(advertised=advertised, busy=busy, embeddings=embeddings):
                engine = self.engine(advertised, busy)
                engine._pack_hint([1, 42], embeddings)
                engine._send.assert_not_called()

    def test_absent_capability_preserves_legacy_protocol(self):
        engine = self.engine()
        engine.info = {}
        engine._pack_hint([1, 42], None)
        engine._send.assert_not_called()

    def test_failed_hint_is_advisory_and_keeps_lock_owner(self):
        engine = self.engine()
        engine._send.side_effect = EngineDied("stopped")
        engine._pack_hint([1, 42], None)
        self.assertTrue(engine.ctl.locked())


if __name__ == "__main__":
    unittest.main()
