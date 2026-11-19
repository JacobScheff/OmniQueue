"""Run the DP itinerary explorer: python Park/plan.py --demo."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import sys
import threading
import webbrowser
from zoneinfo import ZoneInfo

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Park import config
from Park.planner import Context, Ride, plan
from Park.planner.data import WalkingMap, Forecast, demo_forecast, generate_forecast, ROOT

ZONE = ZoneInfo('America/Los_Angeles')


class Explorer:
    def __init__(self, args):
        self.args = args
        self.walks = WalkingMap()
        self.lock = threading.Lock()
        self.cached = {}
        self.fixed = (demo_forecast() if args.demo else
                      json.loads(args.forecast.read_text()) if args.forecast else None)

    def forecast(self, body):
        if self.fixed is not None:
            return self.fixed
        key = (body.get('day', self.args.day), body.get('start', self.args.start))
        if key not in self.cached:
            payload = generate_forecast(*key, self.args.forecast_python, self.args.device)
            if len(self.cached) >= 4:
                self.cached.pop(next(iter(self.cached)))
            self.cached[key] = payload
        return self.cached[key]

    def catalog(self):
        return dict(rides=[dict(id=j, name=r['name'], preference=r['popularity'],
                               x=self.walks.points[j]['x'], y=self.walks.points[j]['y'])
                           for j, r in enumerate(config.RIDES)],
                    paths=[e['geometry'] for e in self.walks.data['edges']],
                    entrance=self.walks.points[-1], demo=self.args.demo,
                    aerial_url=self.walks.aerial_url(),
                    fixed=self.fixed is not None, day=self.args.day or '', start=self.args.start)

    def recommend(self, body):
        payload = self.forecast(body)
        forecast = Forecast(payload)
        n = config.NUM_RIDES
        prefs = body.get('preferences', [r['popularity'] for r in config.RIDES])
        must = set(body.get('must_dos', []))
        done = set(body.get('completed', []))
        location = int(body.get('location', n))
        if len(prefs) != n or not 0 <= location <= n or not (must | done) <= set(range(n)):
            raise ValueError('Invalid preferences, attraction IDs or location')
        now_local = datetime.fromtimestamp(forecast.now, ZONE)
        leave_text = body.get('leave', '23:00')
        clock = datetime.strptime(leave_text, '%H:%M').time()
        leave = datetime.combine(now_local.date(), clock, ZONE)
        if leave <= now_local:
            leave += timedelta(days=1)
        requested_leave = int(leave.timestamp())
        end = min(requested_leave, forecast.end)
        speed, cost = float(body.get('speed', 1.4)), float(body.get('walk_cost', 2))
        budget, beam = float(body.get('seconds', 10)), int(body.get('beam', 200))
        buffer, step = int(body.get('buffer', 5))*60, int(body.get('step', 5))*60
        if (not all(math.isfinite(x) for x in (speed, cost, budget))
                or not .2 <= speed <= 3 or not 0 <= cost <= 50 or not 0.1 <= budget <= 30
                or not 1 <= beam <= 2000 or not 0 <= buffer <= 1800 or step not in (60, 300)):
            raise ValueError('Planner settings out of range')
        # Per-ride deadline overrides are optional explicit user assumptions.
        deadlines = body.get('queue_deadlines', {})
        rides = []
        for j, r in enumerate(config.RIDES):
            close = end
            if str(j) in deadlines:
                clock = datetime.strptime(deadlines[str(j)], '%H:%M').time()
                dt = datetime.combine(now_local.date(), clock, ZONE)
                if dt < now_local.replace(hour=8, minute=0, second=0, microsecond=0):
                    dt += timedelta(days=1)
                close = min(end, int(dt.timestamp()))
            rides.append(Ride(j, r['name'], r['duration_sec'], float(prefs[j]), j in must,
                              j in done, queue_closes=close, finish_by=end))
        context = Context(rides, forecast.now, end, self.walks.matrix(location), forecast.wait_at,
                          speed=speed, walk_cost=cost, buffer=buffer, step=step,
                          beam=beam, seconds=budget, operating_at=forecast.operating_at)
        result = plan(context)
        result['warnings'] = []
        if not result['routes']:
            result['diagnostics'] = [forecast.diagnose(j, forecast.now, end) for j in sorted(must-done)]
            if result['diagnostics'] and result['status'] == 'uncertified':
                result['reason'] = '\n'.join(d['name']+': '+d['reason'] for d in result['diagnostics'])
            elif result['status'] == 'search_incomplete':
                result['reason'] = 'Search budget ended before a complete must-do schedule was found. Increase the search budget.'
        for route in result['routes']:
            previous = location
            for stop in route['stops']:
                stop['path'] = self.walks.paths[previous, stop['ride_id']]
                previous = stop['ride_id']
            route['exit_path'] = self.walks.paths[previous, n]
            route['warnings'] = [f"{s['name']}: model estimates {s['operating_probability']:.0%} "
                                 'operating probability at queue entry.' for s in route['stops']
                                 if s['operating_probability'] < 0.9]
        result.update(now=forecast.now, leave=end, requested_leave=requested_leave,
                      forecast_end=forecast.end, departure_limited_by_forecast=requested_leave > forecast.end,
                      history_as_of=payload['as_of'],
                      demo=bool(payload.get('demo')), location=location,
                      source='Synthetic demonstration' if payload.get('demo') else 'Wait-time model forecast')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--demo', action='store_true', help='Synthetic waits; no ML dependencies')
    source.add_argument('--forecast', type=Path, help='JSON returned by ml.run.predict()')
    parser.add_argument('--day', help='Cached operating day YYYY-MM-DD (default latest)')
    parser.add_argument('--start', default='10:00', help='Observed THROUGH this park-local time')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    local_python = ROOT / 'Wait Time Predictions/.venv/Scripts/python.exe'
    parser.add_argument('--forecast-python', default=str(local_python if local_python.exists() else sys.executable))
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    app = Explorer(args)

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, payload, content_type='application/json'):
            data = json.dumps(payload, allow_nan=False).encode() if content_type == 'application/json' else payload
            self.send_response(status)
            self.send_header('Content-Type', content_type + '; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/api/catalog':
                self.send(200, app.catalog())
            elif self.path == '/':
                self.send(200, (Path(__file__).parent/'planner/explorer.html').read_bytes(), 'text/html')
            else:
                self.send(404, {'error': 'Not found'})

        def do_POST(self):
            if self.path != '/api/plan':
                self.send(404, {'error': 'Not found'})
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 100_000:
                    raise ValueError('Invalid request size')
                body = json.loads(self.rfile.read(length))
                with app.lock:
                    result = app.recommend(body)
                self.send(200, result)
            except (ValueError, KeyError, TypeError, OverflowError) as error:
                self.send(400, {'error': str(error)})
            except Exception as error:
                import traceback
                traceback.print_exc()
                self.send(500, {'error': str(error)})

    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    except PermissionError:
        # Windows can reserve blocks of ports, including the preferred port.
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'DP itinerary explorer: {url}', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
