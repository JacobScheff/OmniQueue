"""Public read API for OmniQueue Forecasts, deployed independently of the collector."""

import gzip
from datetime import date, datetime, timezone
from functools import lru_cache
from hashlib import sha256

from flask import Flask, jsonify, request

from ml.forecast_service import ForecastBusy, ForecastService, ForecastUnavailable
from ml.forecast_store import ForecastStore
from ml.parks import PARKS, get_park


@lru_cache(maxsize=1)
def cloud_service():
    return ForecastService(ForecastStore())


def query_date(name):
    value = request.args.get(name)
    if value is None:
        return None
    parsed = date.fromisoformat(value)
    if value != parsed.isoformat():
        raise ValueError("Enter a date as YYYY-MM-DD.")
    return parsed


def create_app(service=None):
    app = Flask(__name__)
    app.json.sort_keys = False

    def backend():
        return service if service is not None else cloud_service()

    def now():
        return datetime.now(timezone.utc)

    @app.after_request
    def privacy(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.vary.add("Accept-Encoding")
        if response.status_code == 200 and response.calculate_content_length() > 8192 and request.accept_encodings["gzip"] > 0:
            response.set_data(gzip.compress(response.get_data(), compresslevel=3))
            response.headers["Content-Encoding"] = "gzip"
        return response

    @app.errorhandler(ForecastUnavailable)
    def unavailable(error):
        return jsonify(error=str(error)), 404

    @app.errorhandler(ForecastBusy)
    def busy(error):
        return jsonify(error=str(error)), 503, {"Retry-After": "2"}

    @app.errorhandler(500)
    def failed(error):
        return jsonify(error="Forecasts are temporarily unavailable. Please try again."), 500

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/parks")
    def parks():
        return {"parks": [park.metadata() for park in PARKS.values()]}

    @app.get("/v1/parks/<slug>/day")
    def day(slug):
        try:
            park, requested = get_park(slug), query_date("date")
        except ValueError:
            return jsonify(error="Choose a supported park and a valid date."), 400
        response = jsonify(backend().day_payload(park, requested, now()))
        revision = sha256(response.get_data()).hexdigest()
        if request.if_none_match.contains_weak(revision):
            response.set_data(b"")
            response.status_code = 304
        response.set_etag(revision, weak=True)
        return response

    @app.get("/v1/parks/<slug>/dates")
    def dates(slug):
        try:
            park, before = get_park(slug), query_date("before")
        except ValueError:
            return jsonify(error="Choose a supported park and a valid date."), 400
        today = backend().today(park, now())
        upper = min(before, today) if before is not None else today
        rows = backend().store.public_dates(park.slug, upper.isoformat(), 32)
        return {"dates": rows[:31], "next": rows[30]["date"] if len(rows) == 32 else None}

    @app.get("/v1/parks/<slug>/forecast")
    def forecast(slug):
        try:
            park, requested = get_park(slug), query_date("date")
            raw = request.args.get("start")
            cutoff = int(raw) if raw is not None else None
            if cutoff is not None and (not 0 < cutoff < 4_102_444_800 or cutoff % 300):
                raise ValueError()
        except ValueError:
            return jsonify(error="Choose a supported park, valid date, and a five-minute forecast start."), 400
        return jsonify(backend().forecast(park, requested, cutoff, now()))

    return app


app = create_app()
