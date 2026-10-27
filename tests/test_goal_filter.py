import unittest
from rm_rl.waypoint.goal_filter import GoalFilter

A, B, C = (1., 1.), (2., 2.), (3., 3.)

class GoalFilterTests(unittest.TestCase):
    def test_oscillation_and_confirmation(self):
        f = GoalFilter()
        self.assertEqual(f.update(A, 0), A)
        self.assertEqual(f.update(B, .2), A)
        self.assertEqual(f.update(A, .8), A)
        self.assertEqual(f.update(B, 1), A)
        self.assertEqual(f.update(B, 1.9), A)
        self.assertEqual(f.update(B, 2), B)

    def test_hold_and_changed_pending(self):
        f = GoalFilter()
        f.update(A, 0)
        f.update(B, .1)
        self.assertEqual(f.update(B, 1.1), A)
        f.update(C, 1.5)
        self.assertEqual(f.update(C, 2), A)
        self.assertEqual(f.update(C, 2.5), C)

    def test_urgent_reset_and_clock_reset(self):
        f = GoalFilter()
        f.update(A, 0)
        self.assertEqual(f.update(B, .1, urgent=True), B)
        self.assertEqual(f.update(A, .2), B)
        f.reset()
        self.assertEqual(f.update(C, .3), C)
        self.assertEqual(f.update(A, 0), A)

    def test_disabled_and_invalid(self):
        f = GoalFilter(0, 0)
        for i, p in enumerate((A, B, C, A)):
            self.assertEqual(f.update(p, i * .1), p)
        for v in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): GoalFilter(v)
        with self.assertRaises(ValueError): f.update((float('nan'), 0), 2)

if __name__ == '__main__': unittest.main()
