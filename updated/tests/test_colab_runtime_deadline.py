import unittest
from colab.runtime_deadline import stop_reason


class RuntimeDeadlineTests(unittest.TestCase):
    def test_deadline_stops_even_if_run_not_complete(self):
        self.assertEqual(stop_reason({'stage': 'masking'}, 7200, 7200), 'two_hour_deadline')
        self.assertEqual(stop_reason({}, 7201, 7200), 'two_hour_deadline')

    def test_success_requires_a_saved_archive(self):
        self.assertIsNone(stop_reason({'stage': 'exporting'}, 1, 7200))
        self.assertIsNone(stop_reason({'stage': 'complete'}, 1, 7200))
        self.assertEqual(stop_reason(dict(stage='complete', archive='a.zip', archive_sha256='abc'),
                                     1, 7200), 'complete_archive_saved')

    def test_failure_releases_runtime(self):
        self.assertEqual(stop_reason({'stage': 'failed'}, 1, 7200), 'experiment_failed')
