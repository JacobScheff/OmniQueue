// Run with: node --test tests/test_visualizer_errors.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function visualizer() {
  const elements = new Map();
  const element = () => ({
    children: [], value: "", scrollTop: 0, style: {}, html: "",
    addEventListener() {}, setAttribute() {},
    append(...children) { this.children.push(...children); },
    replaceChildren() { this.children = []; },
    insertAdjacentHTML(position, html) { this.html += html; },
  });
  const context = vm.createContext({Intl, Date, Map, Set, document: {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, element());
      return elements.get(id);
    },
    createElement: element, createElementNS: element,
  }});
  const source = fs.readFileSync(path.join(__dirname, "../ml/visualizer/app.js"), "utf8");
  // Load the real UI functions without starting network requests or the map.
  vm.runInContext(source.replace(/\ninit\(\);\s*$/, "\n"), context);
  const ui = vm.runInContext("({state, predictionError, errorBadge, renderList, renderMarkers, renderDetail})", context);
  ui.state.meta = {locations: {}};
  ui.state.day = {timestamps: [100]};
  ui.state.observed.set(100, {source: "observed", rides: []});
  return {...ui, context, elements};
}

const ride = (wait, status = "OPERATING", id = "a") => ({id, name: "Ride A", wait, status});
const frame = {source: "predicted", timestamp: 100};
function actual(ui, row) { ui.state.observed.get(100).rides = [row]; }

test("numeric error uses prediction minus actual, zero waits, and absolute severity thresholds", () => {
  const ui = visualizer();
  for (const [predicted, recorded, label, severity] of [
    [0, 0, "0 min", "small"], [5, 0, "+5 min", "small"], [0, 5, "-5 min", "small"],
    [30, 20, "+10 min", "medium"], [0, 15, "-15 min", "medium"],
    [20, 0, "+20 min", "large"], [0, 300, "-300 min", "large"],
  ]) {
    actual(ui, ride(recorded));
    const error = ui.predictionError(ride(predicted), frame);
    assert.equal(error.label, label);
    assert.equal(error.severity, severity);
    assert.ok(error.title.includes(`recorded ${recorded} min`));
  }
});

test("numeric versus closed/down/refurbishment mismatches are red in both directions", () => {
  const ui = visualizer();
  for (const status of ["CLOSED", "DOWN", "REFURBISHMENT"]) {
    for (const [predicted, recorded] of [[ride(0), ride(null, status)], [ride(null, status), ride(0)]]) {
      actual(ui, recorded);
      const error = ui.predictionError(predicted, frame);
      assert.equal(error.label, "Mismatch");
      assert.equal(error.severity, "large");
    }
    actual(ui, ride(null, status));
    assert.equal(ui.predictionError(ride(null, status), frame).label, "Match");
  }
  actual(ui, ride(null, "DOWN"));
  assert.equal(ui.predictionError(ride(null, "CLOSED"), frame).severity, "large");
});

test("overflow compares tokens without inventing a numeric error", () => {
  const ui = visualizer();
  actual(ui, ride("300+"));
  assert.equal(ui.predictionError(ride("300+"), frame).label, "Match");
  assert.equal(ui.predictionError(ride(300), frame).label, "Mismatch");
  actual(ui, ride(300));
  assert.equal(ui.predictionError(ride("300+"), frame).severity, "large");
});

test("unknown, missing, interpolated, pending and non-predicted values have no error", () => {
  const ui = visualizer();
  for (const status of ["MISSING", "UNKNOWN"]) {
    actual(ui, ride(null, status));
    assert.equal(ui.predictionError(ride(10), frame), null);
    actual(ui, ride(10));
    assert.equal(ui.predictionError(ride(null, status), frame), null);
  }
  for (const source of ["observed", "interpolated", "missing", "break"]) {
    assert.equal(ui.predictionError(ride(10), {...frame, source}), null);
  }
  assert.equal(ui.predictionError(ride(10), {...frame, pending: true}), null);
  ui.state.observed.get(100).source = "interpolated";
  assert.equal(ui.predictionError(ride(10), frame), null);
  ui.state.observed.clear();
  assert.equal(ui.errorBadge(ride(10), frame), "");
});

test("comparison matches actuals by ride ID and exact timestamp", () => {
  const ui = visualizer();
  actual(ui, ride(100, "OPERATING", "other"));
  assert.equal(ui.predictionError(ride(10), frame), null);
  actual(ui, ride(5));
  assert.equal(ui.predictionError(ride(10), {...frame, timestamp: 101}), null);
  assert.equal(ui.predictionError(ride(10), frame).label, "+5 min");
});

test("list, map pins, tooltips and details display the same error and recorded value", () => {
  const ui = visualizer();
  const predicted = ride(30);
  actual(ui, ride(20));
  ui.renderList([predicted], frame);
  const row = ui.elements.get("ride-list").children[0];
  assert.match(row.children[1].html, /prediction-error medium/);
  assert.match(row.children[1].html, /\+10 min/);
  assert.match(row.children[1].html, /recorded 20 min/);

  const marker = {setIcon(icon) { this.icon = icon; }, setTooltipContent(text) { this.tooltip = text; }, setZIndexOffset() {}};
  ui.state.map = {};
  ui.state.meta.locations.a = {latitude: 33.81, longitude: -117.92};
  ui.state.markers.set("a", marker);
  ui.context.L = {divIcon: options => options};
  ui.renderMarkers([predicted], frame);
  assert.match(marker.icon.html, /\+10m/);
  assert.match(marker.tooltip, /recorded 20 min/);

  ui.renderDetail(predicted, frame);
  assert.match(ui.elements.get("detail-wait").html, /\+10 min/);
  assert.match(ui.elements.get("detail-meta").textContent, /recorded 20 min/);
});

test("unavailable forecasts show status and probabilities without numeric wait ranges", () => {
  const ui = visualizer();
  for (const status of ["DOWN", "CLOSED"]) {
    const predicted = {...ride(null, status), down_probability: .6, closed_probability: .3};
    ui.renderDetail(predicted, frame);
    assert.equal(ui.elements.get("detail-wait").textContent, status === "DOWN" ? "Down" : "Closed");
    assert.match(ui.elements.get("detail-meta").textContent, /60% down · 30% closed/);
    assert.doesNotMatch(ui.elements.get("detail-meta").textContent, /min p10/);
  }
});
