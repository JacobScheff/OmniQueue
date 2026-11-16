"use strict";

const $ = id => document.getElementById(id);
const state = {
  meta: null, day: null, observed: new Map(), predicted: new Map(), index: 0,
  cutoff: Infinity, generation: 0, loading: false, forecastError: "", timer: null,
  map: null, markers: new Map(), selected: null, sortWait: false, layers: {},
};
const clockFormatter = new Intl.DateTimeFormat("en-US", {
  timeZone: "America/Los_Angeles", hour: "numeric", minute: "2-digit",
});
const inputFormatter = new Intl.DateTimeFormat("en-GB", {
  timeZone: "America/Los_Angeles", hour: "2-digit", minute: "2-digit", hourCycle: "h23",
});
const dateFormatter = new Intl.DateTimeFormat("en-US", {
  timeZone: "UTC", weekday: "short", month: "short", day: "numeric", year: "numeric",
});
const clock = tick => clockFormatter.format(new Date(tick * 1000));
const clockInput = tick => inputFormatter.format(new Date(tick * 1000));
const escapeHTML = value => String(value).replace(/[&<>"']/g, char => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
})[char]);
const sourceNames = {observed: "Recorded data", predicted: "Model prediction", interpolated: "Interpolated data", missing: "No observation", break: "Scheduled break"};

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
  return result;
}

function message(text, error = false) {
  $("message").hidden = !text;
  $("message").textContent = text;
  $("message").classList.toggle("error", error);
}

function initMap() {
  if (!window.L) {
    message("The map library could not load. Check your internet connection and reload. You can still explore waits in the attraction list.", true);
    return;
  }
  state.map = L.map("map", {
    zoomControl: false, minZoom: 14, maxZoom: 21, preferCanvas: true,
    zoomSnap: 0.25, zoomDelta: 0.25, zoomAnimation: true,
    wheelPxPerZoomLevel: 60, wheelDebounceTime: 40,
  })
    .setView([33.8130, -117.9191], 17);
  L.control.zoom({position: "bottomright"}).addTo(state.map);
  // The county's public image service provides much more detail than the
  // zoom-16 USGS cache. Its export endpoint works even where cached tiles do not.
  const aerialURL = "https://www.ocgis.com/arcpub/rest/services/Aerial_Imagery_Countywide/OC_Aerial_3in_WGS84/ImageServer/exportImage";
  const AerialTiles = L.TileLayer.extend({
    getTileUrl(coords) {
      const halfWorld = Math.PI * 6378137;
      const span = 2 * halfWorld / 2 ** coords.z;
      const left = -halfWorld + coords.x * span;
      const top = halfWorld - coords.y * span;
      const pixels = L.Browser.retina ? 512 : 256;
      const parameters = new URLSearchParams({
        bbox: [left, top - span, left + span, top].join(","),
        bboxSR: "3857", imageSR: "3857", size: `${pixels},${pixels}`,
        format: "jpg", compressionQuality: "90", f: "image",
      });
      return `${aerialURL}?${parameters}`;
    },
  });
  state.layers.aerial = new AerialTiles(aerialURL, {
    maxNativeZoom: 20, maxZoom: 21, updateWhenIdle: true, updateWhenZooming: false,
    keepBuffer: 1,
    attribution: 'Imagery: <a href="https://www.ocgis.com/ocpw/historicalimagery/index.html">County of Orange</a>',
  });
  state.layers.streets = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxNativeZoom: 19, maxZoom: 21,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  });
  let failures = 0;
  state.layers.aerial.on("tileerror", () => {
    if (++failures === 4 && state.map.hasLayer(state.layers.aerial)) {
      selectLayer("streets");
      message("Aerial imagery is unavailable. Showing the street map instead.");
    }
  });
  state.layers.streets.on("tileerror", () => {
    if (state.map.hasLayer(state.layers.streets)) message("Street tiles could not load. Wait markers and the attraction list remain available.");
  });
  state.layers.aerial.addTo(state.map);
  state.map.attributionControl.addAttribution('Locations: <a href="https://themeparks.wiki/">ThemeParks.wiki</a>');
}

function selectLayer(name) {
  if (!state.map) return;
  for (const [key, layer] of Object.entries(state.layers)) {
    if (key === name) layer.addTo(state.map);
    else state.map.removeLayer(layer);
    $(key).classList.toggle("active", key === name);
    $(key).setAttribute("aria-pressed", String(key === name));
  }
}

function frameAt(tick) {
  if (tick >= state.cutoff) return state.predicted.get(tick) || {timestamp: tick, source: "predicted", rides: [], pending: true};
  const observed = state.observed.get(tick);
  if (observed && (!observed.endpoints || observed.endpoints[1] < state.cutoff)) return observed;
  return {timestamp: tick, source: state.day.expectedSet.has(tick) ? "missing" : "break", rides: []};
}

function currentFrame() {
  return frameAt(state.day.timestamps[state.index]);
}

function resolveClock(text) {
  if (!/^\d{2}:\d{2}$/.test(text)) throw new Error("Choose a valid park-local time.");
  const [hour, minute] = text.split(":").map(Number);
  if (hour > 23 || minute > 59) throw new Error("Choose a valid park-local time.");
  const total = (Math.ceil((hour * 60 + minute) / 5) * 5) % 1440;
  const rounded = `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
  const tick = state.day.timestamps.find(value => clockInput(value) === rounded);
  if (tick === undefined) throw new Error(`Choose a time between ${clock(state.day.timestamps[0])} and ${clock(state.day.timestamps.at(-1))}.`);
  return tick;
}

function stopPlayback() {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  $("play").textContent = "▶";
  $("play").setAttribute("aria-label", "Play timeline");
}

function setIndex(index) {
  if (!state.day) return;
  state.index = Math.max(0, Math.min(state.day.timestamps.length - 1, index));
  $("timeline").value = state.index;
  $("view-time").value = clockInput(state.day.timestamps[state.index]);
  render();
}

function prepareStart() {
  const cutoff = resolveClock($("start").value);
  if (cutoff <= state.day.timestamps[0]) throw new Error("Predictions need at least one earlier observation. Choose a start after park opening.");
  $("start").value = clockInput(cutoff);
  state.cutoff = cutoff;
  state.predicted.clear();
  state.forecastError = "";
  return cutoff;
}

async function loadDay(key) {
  const generation = ++state.generation;
  stopPlayback();
  state.loading = false;
  state.selected = null;
  $("detail").hidden = true;
  $("generate").disabled = true;
  $("day-note").textContent = "Loading saved observations…";
  try {
    const day = await api(`/api/day?date=${encodeURIComponent(key)}`);
    if (generation !== state.generation) return;
    state.day = {...day, expectedSet: new Set(day.expected_timestamps)};
    state.observed = new Map(day.snapshots.map(frame => [frame.timestamp, frame]));
    state.predicted.clear();
    state.forecastError = "";
    state.cutoff = Infinity;
    if (!day.timestamps.length) throw new Error("This day has no scheduled ticks to display.");
    $("timeline").max = day.timestamps.length - 1;
    for (const id of ["start", "view-time", "timeline", "play", "back", "forward"]) $(id).disabled = false;
    const zone = new Intl.DateTimeFormat("en-US", {timeZone: "America/Los_Angeles", timeZoneName: "short"})
      .formatToParts(new Date(day.timestamps[0] * 1000)).find(part => part.type === "timeZoneName").value;
    $("zone").textContent = zone;
    const recorded = day.snapshots.filter(frame => frame.source === "observed").length;
    const interpolated = day.snapshots.length - recorded;
    $("day-note").textContent = `${recorded} recorded ticks${interpolated ? ` · ${interpolated} interpolated` : ""} · ${day.expected_timestamps.length} scheduled`;
    if (!$("start").value) $("start").value = state.meta.default_start || "12:00";
    try {
      prepareStart();
    } catch {
      const latest = day.latest_observation;
      const fallback = day.timestamps.find(tick => latest !== null && tick > latest) || day.timestamps[Math.min(48, day.timestamps.length - 1)];
      $("start").value = clockInput(fallback);
      if (day.timestamps.length > 1) prepareStart();
    }
    if (!state.meta.checkpoint_available) {
      state.cutoff = Infinity;
      message("No trained checkpoint found. You can browse recorded waits now. Run python ml/train.py, then reload to enable predictions.");
    } else message(state.meta.location_warning);
    setIndex(Math.max(0, day.timestamps.indexOf(Number.isFinite(state.cutoff) ? state.cutoff : day.latest_observation)));
    $("generate").disabled = !state.meta.checkpoint_available;
    $("generate-label").textContent = "Generate forecast";
    renderTimeline();
    if (state.meta.checkpoint_available && day.timestamps.length > 1) generate();
  } catch (error) {
    if (generation !== state.generation) return;
    message(error.message, true);
    $("day-note").textContent = "Could not load this operating day";
  }
}

async function generate() {
  if (!state.day || !state.meta.checkpoint_available) return;
  let cutoff;
  try { cutoff = prepareStart(); }
  catch (error) { message(error.message, true); return; }
  const generation = ++state.generation;
  state.loading = true;
  $("generate").disabled = true;
  $("generate-label").textContent = "Generating forecast…";
  $("forecast-note").classList.remove("error");
  $("forecast-note").textContent = `Rolling forward from ${clock(cutoff)} · ${state.meta.samples} sampled paths`;
  message(state.meta.location_warning);
  renderTimeline();
  render();
  try {
    const job = await api("/api/forecast", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({date: state.day.date, start: $("start").value}),
    });
    while (generation === state.generation) {
      const result = await api(`/api/forecast?job=${encodeURIComponent(job.job)}`);
      if (generation !== state.generation) return;
      if (result.state === "complete") {
        state.predicted = new Map(result.snapshots.map(frame => [frame.timestamp, frame]));
        state.loading = false;
        $("forecast-note").textContent = `History through ${clock(Date.parse(result.as_of) / 1000)} · ${result.samples} paths. Dashed pins are predictions.`;
        renderTimeline();
        render();
        return;
      }
      $("generate-label").textContent = result.state === "queued" ? "Forecast queued…" : "Generating forecast…";
      await new Promise(resolve => setTimeout(resolve, 650));
    }
  } catch (error) {
    if (generation !== state.generation) return;
    state.forecastError = error.message;
    state.loading = false;
    message(error.message, true);
    $("forecast-note").textContent = "Choose a start with complete earlier history, then generate again.";
    $("forecast-note").classList.add("error");
    render();
  } finally {
    if (generation === state.generation) {
      state.loading = false;
      $("generate").disabled = false;
      $("generate-label").textContent = state.predicted.size ? "Regenerate forecast" : "Generate forecast";
    }
  }
}

function displayWait(ride, unit = false) {
  if (!ride) return "—";
  if (ride.wait !== null && ride.wait !== undefined) return `${ride.wait}${unit ? " min" : ""}`;
  return ({DOWN: "Down", CLOSED: "Closed", REFURBISHMENT: "Refurb", UNKNOWN: "Unknown", MISSING: "No data"})[ride.status] || "—";
}

function render() {
  if (!state.day) return;
  const tick = state.day.timestamps[state.index];
  const frame = currentFrame();
  $("clock").textContent = clock(tick);
  $("timeline").setAttribute("aria-valuetext", `${clock(tick)}, ${frame.pending ? "prediction unavailable" : sourceNames[frame.source]}`);
  const localDay = new Intl.DateTimeFormat("en-CA", {timeZone: "America/Los_Angeles", year: "numeric", month: "2-digit", day: "2-digit"}).format(new Date(tick * 1000));
  $("day-offset").textContent = localDay !== state.day.date ? "+1 day" : "";
  $("source").className = `source ${frame.source}`;
  $("source").textContent = frame.pending ? (state.loading ? "Preparing prediction" : "Prediction unavailable") : sourceNames[frame.source];
  $("source-note").textContent = frame.pending ? (state.forecastError ? "Adjust the prediction start to resolve the error." : "Generate a forecast to explore this time.")
    : frame.source === "predicted" ? `${state.meta.samples} paths · median wait when operating`
    : frame.source === "interpolated" ? `Estimated between ${clock(frame.endpoints[0])} and ${clock(frame.endpoints[1])}`
    : frame.source === "observed" ? "Saved five-minute snapshot · real park observations"
    : frame.source === "break" ? "No operating session scheduled at this time."
    : "No saved observation for this five-minute tick.";
  const byId = new Map(frame.rides.map(ride => [ride.id, ride]));
  const rides = state.day.rides.map(ride => ({...ride, ...(byId.get(ride.id) || {wait: null, status: "MISSING"})}));
  renderList(rides, frame);
  renderMarkers(rides, frame);
  if (state.selected) renderDetail(rides.find(ride => ride.id === state.selected), frame);
  $("back").disabled = state.index === 0;
  $("forward").disabled = state.index === state.day.timestamps.length - 1;
}

function renderList(rides, frame) {
  const search = $("search").value.trim().toLocaleLowerCase();
  const filtered = rides.filter(ride => ride.name.toLocaleLowerCase().includes(search));
  filtered.sort(state.sortWait ? (a, b) => {
    const number = ride => ride.wait === "300+" ? 301 : typeof ride.wait === "number" ? ride.wait : Infinity;
    return number(a) - number(b) || a.name.localeCompare(b.name);
  } : (a, b) => a.name.localeCompare(b.name));
  $("ride-count").textContent = `${filtered.length}`;
  const list = $("ride-list");
  const scroll = list.scrollTop;
  list.replaceChildren();
  for (const ride of filtered) {
    const button = document.createElement("button");
    button.className = `ride-row${state.selected === ride.id ? " selected" : ""}`;
    const label = document.createElement("span");
    const name = document.createElement("span");
    name.className = "ride-name";
    name.textContent = ride.name;
    const subtitle = document.createElement("span");
    subtitle.className = "ride-subtitle";
    subtitle.textContent = !state.meta.locations[ride.id] ? "Location unavailable · view details"
      : frame.pending ? "Prediction unavailable"
      : frame.source === "predicted" ? `Forecast · ${Math.round((ride.down_probability || 0) * 100)}% chance down`
      : ride.status === "MISSING" ? "Missing observation"
      : frame.source === "interpolated" ? "Interpolated wait" : "Recorded wait";
    label.append(name, subtitle);
    const pill = document.createElement("span");
    pill.className = `wait-pill ${frame.source}${ride.wait === null ? " inactive" : ""}`;
    pill.textContent = displayWait(ride);
    if (ride.wait !== null) {
      const unit = document.createElement("small"); unit.textContent = "min"; pill.append(unit);
    }
    button.append(label, pill);
    button.addEventListener("click", () => selectRide(ride.id));
    list.append(button);
  }
  if (!filtered.length) { const empty = document.createElement("p"); empty.className = "empty-list"; empty.textContent = "No attractions match your search."; list.append(empty); }
  list.scrollTop = scroll;
}

function renderMarkers(rides, frame) {
  if (!state.map) return;
  const keep = new Set();
  for (const ride of rides) {
    const point = state.meta.locations[ride.id];
    if (!point) continue;
    keep.add(ride.id);
    const inactive = ride.wait === null;
    const text = frame.pending || ["missing", "break"].includes(frame.source) ? "—" : displayWait(ride);
    const html = `<div class="map-pin ${frame.source}${inactive ? " inactive" : ""}${state.selected === ride.id ? " selected" : ""}">${escapeHTML(text)}</div>`;
    const icon = L.divIcon({className: "wait-marker", html, iconSize: [43, 32], iconAnchor: [21, 35], tooltipAnchor: [0, -35]});
    let marker = state.markers.get(ride.id);
    if (!marker) {
      marker = L.marker([point.latitude, point.longitude], {icon, title: ride.name, alt: ride.name, riseOnHover: true})
        .on("click", () => selectRide(ride.id)).addTo(state.map);
      marker.bindTooltip("", {direction: "top"});
      state.markers.set(ride.id, marker);
    } else marker.setIcon(icon);
    marker.setTooltipContent(`${escapeHTML(ride.name)}<br>${escapeHTML(text)}${!inactive ? " min" : ""} · ${frame.pending ? "Prediction unavailable" : sourceNames[frame.source]}`);
    marker.setZIndexOffset(state.selected === ride.id ? 2000 : inactive ? 0 : 100);
  }
  for (const [id, marker] of state.markers) if (!keep.has(id)) { state.map.removeLayer(marker); state.markers.delete(id); }
}

function selectRide(id) {
  state.selected = id;
  const point = state.meta.locations[id];
  if (state.map && point) state.map.flyTo([point.latitude, point.longitude], Math.max(18, state.map.getZoom()), {duration: .45});
  render();
}

function renderDetail(ride, frame) {
  if (!ride) { $("detail").hidden = true; return; }
  $("detail").hidden = false;
  $("detail-name").textContent = ride.name;
  $("detail-source").textContent = `${frame.pending ? "Prediction unavailable" : sourceNames[frame.source]} · ${clock(frame.timestamp)}`;
  $("detail-source").style.color = frame.source === "predicted" ? "var(--purple)" : frame.source === "interpolated" ? "var(--amber)" : "var(--green)";
  $("detail-wait").textContent = displayWait(ride);
  if (ride.wait !== null) { const unit = document.createElement("small"); unit.textContent = frame.source === "predicted" ? " min · median" : " min"; $("detail-wait").append(unit); }
  $("detail-meta").textContent = frame.pending ? "Generate a forecast for this time."
    : frame.source === "predicted" ? `${ride.p10 ?? "—"}–${ride.p90 ?? "—"} min p10–p90 when operating · ${Math.round((ride.down_probability || 0) * 100)}% chance down`
    : frame.source === "interpolated" ? "Linearly interpolated from surrounding observations; rounded to five minutes."
    : "Waits are saved in five-minute buckets. Gaps in the chart indicate unavailable waits.";
  drawSparkline(ride.id);
}

function drawSparkline(id) {
  const svg = $("sparkline");
  svg.replaceChildren();
  const points = state.day.timestamps.map((tick, index) => {
    const frame = frameAt(tick);
    const ride = frame.rides.find(row => row.id === id);
    return {index, source: frame.source, value: ride?.wait === "300+" ? 300 : typeof ride?.wait === "number" ? ride.wait : null};
  });
  const max = Math.max(30, ...points.map(point => point.value || 0));
  const x = index => 25 + index / Math.max(1, points.length - 1) * 292;
  const y = value => 78 - value / max * 64;
  function element(name, attributes, text) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", name);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    svg.append(node); return node;
  }
  for (const value of [0, max]) {
    element("line", {x1: 25, y1: y(value), x2: 317, y2: y(value), stroke: "#e5eae1", "stroke-width": 1});
    element("text", {x: 0, y: y(value) + 3, fill: "#7a837c", "font-size": 8}, value);
  }
  for (let index = 1; index < points.length; index++) {
    const a = points[index - 1], b = points[index];
    if (a.value === null || b.value === null) continue;
    element("line", {x1: x(a.index), y1: y(a.value), x2: x(b.index), y2: y(b.value),
      stroke: b.source === "predicted" ? "#7750c7" : b.source === "interpolated" ? "#a76b13" : "#147b69",
      "stroke-width": 1.8, ...(b.source === "predicted" ? {"stroke-dasharray": "3 2"} : {})});
  }
  element("line", {x1: x(state.index), y1: 7, x2: x(state.index), y2: 83, stroke: "#183a33", "stroke-width": 1, opacity: .5});
  $("spark-start").textContent = clock(state.day.timestamps[0]);
  $("spark-end").textContent = clock(state.day.timestamps.at(-1));
}

function renderTimeline() {
  const ticks = state.day.timestamps;
  const colors = {observed: "#6db29d", interpolated: "#d9b36f", predicted: "#bb9fe5", missing: "#e1e5de", break: "#f1f2ee"};
  const stops = ticks.flatMap((tick, index) => {
    const color = colors[frameAt(tick).source];
    return [`${color} ${index / ticks.length * 100}%`, `${color} ${(index + 1) / ticks.length * 100}%`];
  });
  $("timeline").style.setProperty("--track", `linear-gradient(to right,${stops.join(",")})`);
  const fraction = (state.cutoff - ticks[0]) / Math.max(300, ticks.at(-1) - ticks[0]);
  $("boundary").hidden = !Number.isFinite(fraction) || fraction < 0 || fraction > 1;
  $("boundary").style.left = `${fraction * 100}%`;
  $("boundary-label").style.transform = fraction > .9 ? "translateX(-35%)" : fraction < .1 ? "translateX(35%)" : "";
  $("boundary-label").textContent = `Forecast ${clock(state.cutoff === Infinity ? ticks.at(-1) : state.cutoff)}`;
  $("timeline-labels").replaceChildren();
  for (let index = 0; index <= 4; index++) {
    const tick = ticks[Math.round(index * (ticks.length - 1) / 4)];
    const label = document.createElement("span"); label.textContent = clock(tick); $("timeline-labels").append(label);
  }
}

$("day").addEventListener("change", () => loadDay($("day").value));
$("generate").addEventListener("click", generate);
$("start").addEventListener("change", () => {
  if (!state.meta?.checkpoint_available) return;
  ++state.generation;
  state.loading = false;
  $("generate").disabled = !state.meta?.checkpoint_available;
  $("generate-label").textContent = "Generate forecast";
  try { prepareStart(); renderTimeline(); render(); message("Prediction start changed. Generate a forecast to update the purple part of the timeline."); }
  catch (error) { message(error.message, true); }
});
$("view-time").addEventListener("change", () => {
  stopPlayback();
  try { setIndex(state.day.timestamps.indexOf(resolveClock($("view-time").value))); }
  catch (error) { message(error.message, true); $("view-time").value = clockInput(state.day.timestamps[state.index]); }
});
$("timeline").addEventListener("input", () => { stopPlayback(); setIndex(Number($("timeline").value)); });
$("back").addEventListener("click", () => { stopPlayback(); setIndex(state.index - 1); });
$("forward").addEventListener("click", () => { stopPlayback(); setIndex(state.index + 1); });
$("play").addEventListener("click", () => {
  if (state.timer) return stopPlayback();
  if (state.index === state.day.timestamps.length - 1) setIndex(0);
  $("play").textContent = "Ⅱ";
  $("play").setAttribute("aria-label", "Pause timeline");
  state.timer = setInterval(() => {
    if (state.index >= state.day.timestamps.length - 1) return stopPlayback();
    setIndex(state.index + 1);
  }, 700);
});
$("search").addEventListener("input", render);
$("sort").addEventListener("click", () => {
  state.sortWait = !state.sortWait;
  $("sort").textContent = state.sortWait ? "Wait ↑" : "Name ↓";
  $("sort").setAttribute("aria-label", state.sortWait ? "Sort attractions by name" : "Sort attractions by wait time");
  render();
});
$("aerial").addEventListener("click", () => selectLayer("aerial"));
$("streets").addEventListener("click", () => selectLayer("streets"));
$("recenter").addEventListener("click", () => { if (state.map) state.map.flyTo([33.8130, -117.9191], 17, {duration: .5}); });
$("close-detail").addEventListener("click", () => { state.selected = null; $("detail").hidden = true; render(); });

async function init() {
  initMap();
  try {
    state.meta = await api("/api/meta");
    if (!state.meta.days.length) throw new Error("No cached days yet. Run python ml/data.py, then reload this page.");
    const select = $("day"); select.replaceChildren();
    for (const day of state.meta.days) {
      const option = document.createElement("option");
      option.value = day.date;
      option.textContent = dateFormatter.format(new Date(`${day.date}T12:00:00Z`));
      select.append(option);
    }
    select.value = state.meta.default_day;
    select.disabled = false;
    $("start").value = state.meta.default_start || "12:00";
    await loadDay(select.value);
  } catch (error) {
    message(error.message, true);
    $("source").textContent = "Archive unavailable";
    $("day-note").textContent = error.message;
    $("forecast-note").textContent = "Export your data and reload to get started.";
  }
}
init();
