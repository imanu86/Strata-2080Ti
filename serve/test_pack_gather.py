"""STRATA_PACK_CONCURRENT's collection window (serve/server.py): the request that took the control lines holds its own
prompt line for a few ms so the prompts the others are encoding arrive as BHINT ahead of it and are read in one packed
pass.  Off (unset / 0) nothing here runs and the admission path is the release one."""
import threading
import time
import unittest
from unittest.mock import Mock, patch

from serve import server
from serve.server import StrataEngine


class PackGather(unittest.TestCase):
    def engine(self, slots=3, pending=0):
        engine = StrataEngine.__new__(StrataEngine)
        engine.info = {"pack_hints": str(slots)} if slots else {}
        engine.ctl = threading.Lock()
        engine.ctl.acquire()                      # this request holds the control lines
        engine.slot_cv = threading.Condition()
        engine.pack_pending = pending
        engine.pack_gathered = 0
        engine._send = Mock()
        engine.alive = lambda: True
        return engine

    def window(self, ms=200):
        return patch.multiple(server, PACK_CONCURRENT=True, PACK_WAIT_MS=ms)

    def test_off_switch_changes_nothing(self):
        engine = self.engine()
        with patch.object(server, "PACK_CONCURRENT", False), patch.object(server, "PACK_WAIT_MS", 0):
            t0 = time.monotonic()
            self.assertEqual(engine._pack_gather(), 0)
            self.assertLess(time.monotonic() - t0, 0.05)     # no window: the release path's timing
        self.assertEqual(engine.pack_gathered, 0)

    def test_no_pack_slots_no_waiting(self):
        engine = self.engine(slots=0)
        with self.window():
            t0 = time.monotonic()
            self.assertEqual(engine._pack_gather(), 0)
            self.assertLess(time.monotonic() - t0, 0.05)     # hints are never sent here: waiting is pointless

    def test_window_collects_the_prompt_that_arrives_late(self):
        engine = self.engine(slots=1)             # the single pack slot is enough: the window ends right there
        late = threading.Thread(target=lambda: (time.sleep(0.02), engine._pack_hint([5, 6, 7], None)))
        late.start()
        with self.window():
            t0 = time.monotonic()
            got = engine._pack_gather()
            spent = time.monotonic() - t0
        late.join()
        self.assertEqual(got, 1)
        self.assertLess(spent, 0.15)              # left as soon as the companion's prompt was ahead (< 200 ms)
        engine._send.assert_called_once_with("BHINT 5,6,7")

    def test_a_half_full_window_is_paid_for_the_rest_of_the_slots(self):
        engine = self.engine(slots=3)             # 1 of 3 slots fed: it waits the window for the others (measured,
        late = threading.Thread(target=lambda: (time.sleep(0.01), engine._pack_hint([5, 6, 7], None)))
        late.start()                              # not estimated: STRATA_PACK_WAIT_MS is the knob)
        with self.window(120):
            t0 = time.monotonic()
            got = engine._pack_gather()
            spent = time.monotonic() - t0
        late.join()
        self.assertEqual(got, 1)
        self.assertGreaterEqual(spent, 0.11)
        self.assertLess(spent, 0.3)

    def test_window_ends_at_the_pack_slots(self):
        engine = self.engine(slots=2)
        with self.window():
            for ids in ([1, 2], [3, 4]):          # as many prompts ahead as the engine can pack at once
                engine._pack_hint(ids, None)
            t0 = time.monotonic()
            got = engine._pack_gather()
            spent = time.monotonic() - t0
        self.assertEqual(got, 2)
        self.assertLess(spent, 0.05)              # a third would wait for the next read anyway
        self.assertEqual(engine.pack_gathered, 2)

    def test_a_lone_reader_pays_the_window_once_and_counts_nothing(self):
        engine = self.engine()
        with patch.object(server, "PACK_CONCURRENT", True), patch.object(server, "PACK_WAIT_MS", 60):
            t0 = time.monotonic()
            got = engine._pack_gather()
            spent = time.monotonic() - t0
        self.assertEqual(got, 0)
        self.assertGreaterEqual(spent, 0.05)      # it did hold the line the window asked for
        self.assertLess(spent, 0.2)               # and no longer
        self.assertEqual(engine.pack_gathered, 0)

    def test_the_window_is_consumed_once(self):
        engine = self.engine(slots=1, pending=1)
        with self.window(1000):                   # every slot is already fed: no wait, and the prompts belong
            t0 = time.monotonic()                 # to the read sent next, not to a second one
            self.assertEqual(engine._pack_gather(), 1)
            self.assertLess(time.monotonic() - t0, 0.05)
            self.assertEqual(engine.pack_pending, 0)
            self.assertEqual(engine.pack_gathered, 1)

    def test_hint_counting_is_off_without_the_switch(self):
        engine = self.engine()
        with patch.object(server, "PACK_CONCURRENT", False):
            engine._pack_hint([1, 42], None)
        engine._send.assert_called_once_with("BHINT 1,42")   # the advisory line is the release one
        self.assertEqual(engine.pack_pending, 0)             # nothing was counted, no Condition touched


if __name__ == "__main__":
    unittest.main()
