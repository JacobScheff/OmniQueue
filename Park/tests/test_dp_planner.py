"""Deterministic routing regressions; no native extension or ML needed."""
import math
import unittest

from Park.planner import Context, Ride, plan


class PlannerTests(unittest.TestCase):
    def context(self, rides, now=7200, leave=8400, wait=None, **settings):
        n = len(rides)
        return Context(rides, now, leave, [[0.0]*(n+2) for _ in range(n+2)],
                       wait or (lambda j, t: 0), buffer=0, step=60,
                       seconds=2, beam=200, **settings)

    def test_late_start_and_must_do_survives_optional_detour(self):
        rides = [Ride(0, 'Required', 120, 0, True, queue_closes=7440),
                 Ride(1, 'Tempting', 600, 100)]
        result = plan(self.context(rides))
        self.assertTrue(result['routes'])
        for route in result['routes']:
            ids = [s['ride_id'] for s in route['stops']]
            self.assertIn(0, ids)
            self.assertEqual(ids[0], 0)
            self.assertGreaterEqual(route['stops'][0]['depart'], 7200)

    def test_wait_action_reaches_reopening(self):
        result = plan(self.context([Ride(0, 'Reopens', 60, 1, True)],
                                  wait=lambda j, t: None if t < 7560 else 0))
        self.assertEqual(result['routes'][0]['stops'][0]['join'], 7560)

    def test_exit_travel_reserved(self):
        ctx = self.context([Ride(0, 'Too late', 1100, 1, True)])
        ctx.distances[0][-1] = 200
        self.assertFalse(plan(ctx)['routes'])

    def test_completed_required_excluded_and_disconnected_rejected(self):
        ctx = self.context([Ride(0, 'Done', 60, 1, True, True),
                            Ride(1, 'Disconnected', 60, 100)])
        ctx.distances[2][1] = math.inf
        result = plan(ctx)
        self.assertEqual(result['required_count'], 0)
        self.assertEqual(result['routes'][0]['stops'], [])

    def test_wait_queried_at_queue_entry_and_queue_cutoff(self):
        seen = []
        ctx = self.context([Ride(0, 'Required', 60, 1, True, queue_closes=7500)],
                           wait=lambda j, t: seen.append(t) or 0)
        ctx.distances[1][0] = 100
        result = plan(ctx)
        self.assertEqual(result['routes'][0]['stops'][0]['join'], 7320)
        self.assertNotIn(7200, seen)

    def test_walking_cost_changes_choice(self):
        rides = [Ride(0, 'Near', 600, 90), Ride(1, 'Far', 600, 100)]
        ctx = self.context(rides, leave=8100, walk_cost=0)
        ctx.distances[2][1] = ctx.distances[1][3] = 140
        self.assertEqual(plan(ctx)['routes'][0]['stops'][0]['ride_id'], 1)
        ctx.walk_cost = 100
        self.assertEqual(plan(ctx)['routes'][0]['stops'][0]['ride_id'], 0)

    def test_matches_exhaustive_tiny_instance(self):
        from itertools import permutations
        rides = [Ride(j, str(j), 180, p) for j, p in enumerate([1, 2, 3])]
        ctx = self.context(rides, leave=7800)
        result = plan(ctx)
        mass = sum(r.preference**2 for r in rides)
        scores = []
        for length in range(4):
            for seq in permutations(range(3), length):
                ready, score = ctx.now, 0
                for j in seq:
                    ready += 180
                    score += 100*rides[j].preference**2/mass*math.exp(-.15*(ready-ctx.now)/54000)
                if ready <= ctx.leave:
                    scores.append(score)
        self.assertAlmostEqual(result['routes'][0]['score'], max(scores), places=3)

    def test_extra_low_preference_ride_is_not_rejected_to_save_an_hour(self):
        ctx = self.context([Ride(0, 'Favorite', 60, 100), Ride(1, 'Extra ride', 3600, 5)],
                           leave=11700, walk_cost=0)
        result = plan(ctx)
        self.assertEqual([s['ride_id'] for s in result['routes'][0]['stops']], [0, 1])

    def test_waiting_and_exit_duration_have_no_hourly_score_deduction(self):
        for required in (False, True):
            with self.subTest(required=required):
                ctx = self.context([Ride(0, 'Reopens', 60, 1, required)],
                                   wait=lambda j, t: None if t < 7560 else 0, walk_cost=0)
                # Charge no distance cost; exit travel must still fit the deadline.
                ctx.distances[0][-1] = 140
                route = plan(ctx)['routes'][0]
                self.assertEqual(route['stops'][0]['join'], 7560)
                self.assertGreater(route['exit_walk_sec'], 0)
                expected = 100 * math.exp(-.15 * (route['stops'][0]['finish']-ctx.now)/54000)
                self.assertAlmostEqual(route['score'], expected, places=3)

    def test_forecast_origin_gap_status_and_no_extrapolation(self):
        from Park.planner.data import Forecast, KNOWN_ENTITY_IDS
        def row(value):
            return dict(id=KNOWN_ENTITY_IDS[0], name='Rise', wait_minutes={'p90': value})
        payload = dict(as_of='2026-11-18T18:00:00+00:00', snapshots=[
            dict(time='2026-11-18T18:05:00+00:00', rides=[row(20)]),
            dict(time='2026-11-18T18:15:00+00:00', rides=[row('300+')])])
        f = Forecast(payload)
        self.assertIsNone(f.wait_at(0, f.now))
        self.assertEqual(f.wait_at(0, f.now+300), 1200)
        self.assertIsNone(f.wait_at(0, f.now+600))
        self.assertIsNone(f.wait_at(0, f.now+900))
        self.assertIsNone(f.wait_at(0, f.now+1200))

    def test_explorer_exposes_forecast_cap_on_requested_departure(self):
        from argparse import Namespace
        from Park.plan import Explorer
        app = Explorer(Namespace(demo=True, forecast=None, day=None, start='10:00'))
        app.fixed['closing_timestamp'] -= 300  # coverage ends at 10:55pm
        result = app.recommend(dict(leave='23:50', completed=list(range(34)), seconds=1))
        self.assertTrue(result['departure_limited_by_forecast'])
        self.assertEqual(result['forecast_end'], result['leave'])
        self.assertEqual(result['requested_leave'] - result['leave'], 55*60)
        result = app.recommend(dict(leave='21:00', completed=list(range(34)), seconds=1))
        self.assertFalse(result['departure_limited_by_forecast'])
        self.assertEqual(result['requested_leave'], result['leave'])

    def test_verified_walks_and_aerial_projection(self):
        from Park.planner.data import WalkingMap
        from urllib.parse import urlparse, parse_qs
        walks = WalkingMap()
        self.assertGreater(walks.distances[-1][0], 0)
        self.assertEqual(len(walks.matrix(0)), 36)
        bbox = list(map(float, parse_qs(urlparse(walks.aerial_url()).query)['bbox'][0].split(',')))
        self.assertLess(bbox[0], -117.92)
        self.assertGreater(bbox[2], -117.916)
        self.assertLess(bbox[1], 33.81)
        self.assertGreater(bbox[3], 33.815)


    def test_single_must_do_below_ninety_percent_still_fits(self):
        from Park.planner.data import Forecast, KNOWN_ENTITY_IDS
        payload = dict(as_of='2026-11-18T18:00:00+00:00', snapshots=[
            dict(time=f'2026-11-18T18:{minute:02d}:00+00:00', rides=[
                dict(id=KNOWN_ENTITY_IDS[0], name='Rise', operating_probability=.8,
                     wait_minutes={'p90': 1})]) for minute in (0, 5, 10, 15, 20)])
        forecast = Forecast(payload)
        ctx = self.context([Ride(0, 'Required', 60, 1, True)],
                           now=forecast.now, leave=forecast.end, wait=forecast.wait_at)
        ctx.operating_at = forecast.operating_at
        result = plan(ctx)
        self.assertTrue(result['routes'])
        self.assertEqual(result['routes'][0]['stops'][0]['operating_probability'], .8)
        # Reproduce the previous false negative with the old cutoff.
        ctx.wait_at = Forecast(payload, min_operating=.9).wait_at
        self.assertFalse(plan(ctx)['routes'])

    def test_risk_cost_prefers_more_reliable_equivalent_ride(self):
        ctx = self.context([Ride(0, 'Less reliable', 600, 1), Ride(1, 'More reliable', 600, 1)],
                           leave=8100)
        ctx.operating_at = lambda j, t: .6 if j == 0 else .99
        result = plan(ctx)
        self.assertEqual(result['routes'][0]['stops'][0]['ride_id'], 1)

    def test_probability_relaxation_does_not_invent_waits_for_closed_rides(self):
        from Park.planner.data import Forecast, KNOWN_ENTITY_IDS
        payload = dict(as_of='2026-11-18T18:00:00+00:00', snapshots=[
            dict(time='2026-11-18T18:00:00+00:00', rides=[
                dict(id=KNOWN_ENTITY_IDS[0], name='Rise', status='DOWN',
                     operating_probability=.4, wait_minutes=None)])])
        forecast = Forecast(payload)
        self.assertIsNone(forecast.wait_at(0, forecast.now))
        self.assertEqual(forecast.diagnose(0, forecast.now, forecast.end)['code'], 'unavailable_forecast')
        self.assertEqual(forecast.diagnose(1, forecast.now, forecast.end)['code'], 'missing_forecast')


if __name__ == '__main__':
    unittest.main()
