"""OSM walk routing and adapters for the existing forecast JSON format."""
from bisect import bisect_right
from datetime import datetime
import heapq
import json
import math
from pathlib import Path
import subprocess
from urllib.parse import urlencode

from Park import config
from Park.companion.server.ride_map import KNOWN_ENTITY_IDS, build_config_name_index, resolve_ride_index

ROOT = Path(__file__).resolve().parents[2]


class WalkingMap:
    """Shortest verified paths from the committed OSM graph, with no air-line fallback."""
    def __init__(self):
        self.data = json.loads((ROOT / 'Park/data/pathways.json').read_text())
        self.points = list(self.data['rides']) + [self.data['hubs']['entrance']]
        self.graph = {node: [] for node in self.data['nodes']}
        self.geometry = {}
        for e in self.data['edges']:
            self.graph[e['u']].append((e['v'], e['length_m']))
            self.graph[e['v']].append((e['u'], e['length_m']))
            self.geometry[e['u'], e['v']] = e['geometry']
            self.geometry[e['v'], e['u']] = e['geometry'][::-1]
        self.paths = {}
        self.distances = []
        scale = self.data['meta']['meters_to_display_scale']
        spurs = [math.hypot(p['x']-self.data['nodes'][p['snap_node']]['x'],
                            p['y']-self.data['nodes'][p['snap_node']]['y'])/scale for p in self.points]
        for i, point in enumerate(self.points):
            source = point['snap_node']
            dist, parent, queue = {source: 0.0}, {}, [(0.0, source)]
            while queue:
                cost, u = heapq.heappop(queue)
                if cost != dist[u]:
                    continue
                for v, length in self.graph[u]:
                    candidate = cost + length
                    if candidate < dist.get(v, math.inf):
                        dist[v], parent[v] = candidate, u
                        heapq.heappush(queue, (candidate, v))
            row = []
            for j, target in enumerate(self.points):
                v = target['snap_node']
                row.append(0.0 if i == j else dist.get(v, math.inf)+spurs[i]+spurs[j])
                if v not in dist:
                    continue
                nodes = [v]
                while nodes[-1] != source:
                    nodes.append(parent[nodes[-1]])
                nodes.reverse()
                poly = [[point['x'], point['y']]]
                for u, v in zip(nodes, nodes[1:]):
                    poly.extend(self.geometry[u, v])
                poly.append([target['x'], target['y']])
                self.paths[i, j] = poly
            self.distances.append(row)

    def matrix(self, location):
        indices = list(range(config.NUM_RIDES)) + [location, config.NUM_RIDES]
        return [[self.distances[i][j] for j in indices] for i in indices]

    def aerial_url(self):
        """Invert the exporter's display projection to align a WGS84 image."""
        m = self.data['meta']
        scale = m['meters_to_display_scale']
        lat_m = 111320.0
        lon_m = lat_m * math.cos(math.radians((m['lat_min']+m['lat_max'])/2))
        width = (m['lon_max']-m['lon_min'])*lon_m*scale
        height = (m['lat_max']-m['lat_min'])*lat_m*scale
        left, top = (1000-width)/2, (1000-height)/2
        west = m['lon_min']-left/(scale*lon_m)
        east = m['lon_min']+(1000-left)/(scale*lon_m)
        north = m['lat_max']+top/(scale*lat_m)
        south = m['lat_max']-(1000-top)/(scale*lat_m)
        params = dict(bbox=f'{west},{south},{east},{north}', bboxSR=4326,
                      imageSR=4326, size='2048,2048', format='jpg', f='image',
                      adjustAspectRatio='false')
        return ('https://www.ocgis.com/arcpub/rest/services/Aerial_Imagery_Countywide/'
                'OC_Aerial_3in_WGS84/ImageServer/exportImage?' + urlencode(params))


def timestamp(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('Forecast timestamps must include a timezone')
    return int(dt.timestamp())


class Forecast:
    def __init__(self, payload, quantile='p90', min_operating=0.0):
        self.payload = payload
        self.now = timestamp(payload['as_of'])
        snapshots = list(payload['snapshots'])
        if payload.get('origin_snapshot'):
            snapshots.insert(0, payload['origin_snapshot'])
        self.times = [timestamp(s['time']) for s in snapshots]
        if not self.times or any(a >= b for a, b in zip(self.times, self.times[1:])):
            raise ValueError('Forecast snapshots must be nonempty and strictly increasing')
        self.end = int(payload.get('closing_timestamp', self.times[-1]))
        self.quantile, self.min_operating = quantile, min_operating
        names = build_config_name_index()
        self.rows = []
        for snapshot in snapshots:
            rows = {}
            for row in snapshot['rides']:
                j = resolve_ride_index(row.get('id'), row.get('name', ''), names)
                if j is not None:
                    if j in rows:
                        raise ValueError(f'Ambiguous forecast mapping for {config.RIDES[j]["name"]}')
                    rows[j] = row
            self.rows.append(rows)

    def row_at(self, j, moment):
        index = bisect_right(self.times, moment) - 1
        if index < 0 or moment > self.end or moment - self.times[index] >= 300:
            return None
        return self.rows[index].get(j)

    def operating_at(self, j, moment):
        row = self.row_at(j, moment)
        if row is None:
            return None
        value = row.get('operating_probability', 1)
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1 else None

    def wait_at(self, j, moment):
        row = self.row_at(j, moment)
        probability = self.operating_at(j, moment)
        if (not row or row.get('status') not in (None, 'OPERATING')
                or probability is None or probability <= 0 or probability < self.min_operating):
            return None
        waits = row.get('wait_minutes')
        value = waits.get(self.quantile) if isinstance(waits, dict) else waits
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            return None
        return math.ceil(value * 60)

    def diagnose(self, j, start, end):
        """Separate absent/unavailable forecasts from schedule conflicts."""
        times = [t for t in self.times if start <= t <= end]
        present = [self.row_at(j, t) for t in times]
        rows = [row for row in present if row is not None]
        usable = sum(self.wait_at(j, t) is not None for t in times)
        if not rows:
            reason = 'No forecast coverage for this attraction in the selected time window.'
            code = 'missing_forecast'
        elif usable == 0:
            statuses = sorted({str(row.get('status', 'OPERATING')) for row in rows})
            reason = 'No usable operating wait forecast in this window (statuses: '+', '.join(statuses)+').'
            code = 'unavailable_forecast'
        else:
            reason = (f'{usable} operating forecast ticks exist, but walking, waits, buffers, '
                      'entry deadlines or the combined must-do schedule do not fit. Try a one-minute grid.')
            code = 'schedule_conflict'
        return dict(ride_id=j, name=config.RIDES[j]['name'], code=code,
                    usable_ticks=usable, reason=reason)


def generate_forecast(day, start, python, device='auto'):
    """Isolate the spaced ML project's imports and interpreter from the planner.

    start is the last OBSERVED clock time, not the first predicted tick.
    """
    script = r'''
import contextlib, json, sys
from datetime import datetime, timedelta
from types import SimpleNamespace
from ml import config
from ml.data import load_dataset, PARK_TIMEZONE, interpolate_day
from ml.attractions import modeling_day
from ml.run import predict, summarize, park_timestamp
import torch
day_key, start, device = sys.argv[1:]
dataset = load_dataset(config.DATASET)
days = {**dataset.get('incomplete_days', {}), **dataset['days']}
key = day_key or max(days)
day = interpolate_day(modeling_day(days[key]))
origin = park_timestamp(key, start, day['expected_timestamps'][0])
next_tick = datetime.fromtimestamp(origin, PARK_TIMEZONE)+timedelta(minutes=5)
settings = SimpleNamespace(**{k:getattr(config,k) for k in dir(config) if k.isupper()})
settings.PREDICT_DAY = key
settings.PREDICT_START_TIME = next_tick.strftime('%H:%M')
settings.DEVICE = device
settings.STEPS = None
with contextlib.redirect_stdout(sys.stderr):
    result = predict(settings, print_output=False)
idx = day['timestamps'].index(origin)
result['origin_snapshot'] = summarize(torch.tensor([[day['tokens'][idx]]]), day['rides'], [origin])[0]
result['closing_timestamp'] = day['expected_timestamps'][-1]
print(json.dumps(result))
'''
    result = subprocess.run([str(python), '-c', script, day or '', start, device],
                            cwd=ROOT / 'Wait Time Predictions', capture_output=True,
                            text=True, timeout=300)
    if result.returncode:
        raise ValueError('Forecast failed: ' + result.stderr[-2500:])
    return json.loads(result.stdout)


def demo_forecast():
    """Clearly synthetic, reproducible wait curves for an offline demonstration."""
    from datetime import timezone
    now = int(datetime(2026, 11, 18, 18, tzinfo=timezone.utc).timestamp())  # 10am PST
    snapshots = []
    for tick in range(157):
        rows = []
        for j, ride in enumerate(config.RIDES):
            minutes = max(5, round((ride['popularity']/5 + 16*math.sin(tick/12+j))/5)*5)
            rows.append(dict(id=KNOWN_ENTITY_IDS[j], name=ride['name'],
                             wait_minutes=dict(p50=minutes, p90=minutes+10), operating_probability=1))
        snapshots.append(dict(time=datetime.fromtimestamp(now+tick*300, timezone.utc).isoformat(), rides=rows))
    return dict(as_of=snapshots[0]['time'], snapshots=snapshots,
                closing_timestamp=now+13*3600, demo=True)
