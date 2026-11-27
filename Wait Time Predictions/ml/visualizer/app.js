"use strict";
const $ = id => document.getElementById(id);
const state = {
  meta: null, day: null, observed: new Map(), predicted: new Map(), index: 0,
  cutoff: Infinity, generation: 0, loading: false, forecastError: "", timer: null,
  map: null, markers: new Map(), selected: null, sortWait: false, layers: {},
  dates: [], daysByDate: new Map(), selectedDate: null, calendarMonth: null,
  models: [], selectedModel: null, modelPreview: null, catalogPark: null, loadedPark: null, libraryGeneration: 0,
  jobBusy: false, submitting: false, handledJob: null, jobId: null, stopping: false,
  liveAllSubmitting: false, liveAllBusy: false, liveAllHandled: null,
};
let clockFormatter = new Intl.DateTimeFormat("en-US", {
  timeZone: "America/Los_Angeles", hour: "numeric", minute: "2-digit",
});
let inputFormatter = new Intl.DateTimeFormat("en-GB", {
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
const monthNames = Array.from({length: 12}, (_, month) => new Intl.DateTimeFormat("en-US", {timeZone: "UTC", month: "long"}).format(new Date(Date.UTC(2020, month, 1))));
const dateObject = key => new Date(`${key}T12:00:00Z`);
const shortDate = key => new Intl.DateTimeFormat("en-US", {timeZone: "UTC", month: "short", day: "numeric", year: "numeric"}).format(dateObject(key));
const monthNumber = key => Number(key.slice(0, 4)) * 12 + Number(key.slice(5, 7)) - 1;
const monthKey = number => `${Math.floor(number / 12)}-${String(number % 12 + 1).padStart(2, "0")}`;
function archiveMonths(days) {
  if (!days.length) return [];
  const ordered = [...days].sort((a, b) => a.date.localeCompare(b.date));
  const result = [];
  for (let value = monthNumber(ordered[0].date); value <= monthNumber(ordered.at(-1).date); value++) {
    const key = monthKey(value);
    const entries = ordered.filter(day => day.date.startsWith(key));
    result.push({key, total: entries.length, complete: entries.filter(day => day.complete).length,
      partial: entries.filter(day => !day.complete).length,
      calendarDays: new Date(Date.UTC(Math.floor(value / 12), value % 12 + 1, 0)).getUTCDate()});
  }
  return result;
}

function setPanel(panel, focus = false) {
  for (const name of ["forecast", "manage"]) {
    const active = name === panel;
    $(`${name}-tab`).setAttribute("aria-selected", String(active));
    $(`${name}-tab`).tabIndex = active ? 0 : -1;
    $(`${name}-panel`).hidden = !active;
  }
  $("sidebar-content").scrollTop = 0;
  $("model-workspace").hidden = panel !== "manage";
  $("map-stage").hidden = panel === "manage";
  $("map-timeline").hidden = panel === "manage";
  $("workspace").classList.toggle("is-managing", panel === "manage");
  if (panel === "manage") {
    stopPlayback();
    if (state.park && state.meta) loadModels().catch(error => message(error.message, true));
  } else if (state.map) state.map.invalidateSize();
  if (focus) $(`${panel}-tab`).focus();
}

function splitValues() {
  return ["split-train", "split-validation", "split-test"].map(id => Number($(id).value));
}

function selectedModel() {
  return state.models.find(model => model.id === state.selectedModel);
}

function syncModelButtons() {
  const busy = state.jobBusy || state.submitting;
  const model = selectedModel();
  for (const id of ["refresh-data", "rebuild-data", "live-data"]) $(id).disabled = busy;
  $("train-model").disabled = busy || !state.modelPreview?.ready || !$("run-name").value.trim();
  $("load-model").disabled = busy || !model?.ready;
  $("save-model").disabled = busy || !model?.ready || !$("copy-name").value.trim();
  $("test-model").disabled = busy || !model?.test_ready;
  $("delete-model").disabled = busy || !model?.can_delete;
  for (const button of document.querySelectorAll(".restore-model")) button.disabled = busy || button.dataset.available !== "true";
  $("test-model").title = model?.test_ready ? `Evaluate this model's ${model.test_days} held-out days` : model?.error || "Select a model with a saved test split";
  for (const id of ["model-slot", "run-name", "epochs", "split-train", "split-validation", "split-test", "use-after-train"]) $(id).disabled = busy;
}

function renderSplitPreview(preview) {
  state.modelPreview = preview;
  $("split-bar").replaceChildren();
  if (!preview.ready) {
    $("split-preview").textContent = preview.error;
    $("split-preview").classList.add("model-error");
    $("split-date-list").replaceChildren();
  } else {
    $("split-preview").classList.remove("model-error");
    const validation = preview.groups.find(group => group.name === "validation")?.count > 0;
    $("save-behavior").textContent = validation ? "Best validation model + latest epoch" : "Final epoch (validation disabled)";
    $("train-save-note").textContent = validation
      ? "Automatically saved under this name. Earlier models stay saved. Training starts fresh; validation selects the best epoch."
      : "Automatically saves the final epoch under this name. Earlier models stay saved. Training starts fresh.";
    $("split-preview").textContent = preview.groups.map(group =>
      `${group.count} ${group.name}${group.first_date ? ` (spans ${shortDate(group.first_date)} – ${shortDate(group.last_date)})` : ""}`
    ).join(" · ");
    $("split-seed").textContent = `View exact date assignments · random seed ${preview.seed}`;
    $("split-date-list").replaceChildren();
    for (const group of preview.groups) {
      const label = document.createElement("strong");
      label.textContent = `${group.name} · ${group.count} days`;
      const dates = document.createElement("p");
      dates.textContent = group.dates?.length ? group.dates.join(", ") : "Disabled";
      $("split-date-list").append(label, dates);
    }
    preview.percentages.forEach(percent => { const span = document.createElement("span"); span.style.width = `${percent}%`; $("split-bar").append(span); });
  }
  syncModelButtons();
}

async function loadModels(preferred = state.selectedModel) {
  const park = state.park;
  const generation = ++state.libraryGeneration;
  const [train, validation, test] = splitValues();
  const catalog = await api(`/api/models?train=${train}&validation=${validation}&test=${test}`);
  if (park !== state.park || generation !== state.libraryGeneration) return;
  state.models = catalog.models;
  state.catalogPark = park;
  renderDeletedModels(catalog.deleted || []);
  $("storage-paths").replaceChildren();
  for (const [label, path] of Object.entries(catalog.storage || {})) {
    const name = document.createElement("dt"); name.textContent = label;
    const value = document.createElement("dd"); value.textContent = path;
    $("storage-paths").append(name, value);
  }
  const select = $("model-slot"); select.replaceChildren();
  for (const model of state.models) {
    const option = document.createElement("option");
    option.value = model.id;
    const saved = new Intl.DateTimeFormat("en-US", {timeZone: state.meta.timezone, month: "short", day: "numeric", hour: "numeric", minute: "2-digit"}).format(new Date(model.modified_at * 1000));
    option.textContent = `${model.active ? model.ready ? "Running · " : "Selected · " : ""}${model.name} · ${model.variant === "latest" ? "latest epoch" : "best"} · ${saved}${!model.ready ? " · retrain required" : ""}`;
    select.append(option);
  }
  state.selectedModel = state.models.some(model => model.id === preferred) ? preferred : catalog.active_model || state.models[0]?.id || null;
  if (!state.models.length) { const option = document.createElement("option"); option.textContent = "No saved models yet — create your first model"; option.value = ""; select.append(option); }
  select.value = state.selectedModel || "";
  if (!$("run-name").value.trim()) $("run-name").value = catalog.default_name;
  const active = state.models.find(model => model.active);
  $("active-model-name").textContent = active ? `${active.name}${active.ready ? "" : " — retraining required"}` : "No saved model is running";
  $("studio-title").textContent = `Models for ${state.meta?.park.name || "this park"}`;
  if (active?.ready) { $("forecast-model-badge").textContent = active.name; $("forecast-model-badge").title = active.name; }
  renderSplitPreview(catalog.preview);
  await renderSavedModel();
}

async function renderSavedModel() {
  const model = selectedModel();
  const info = $("selected-model-info");
  if (!model) {
    info.textContent = "Create a named model above to get started.";
    $("delete-model-note").textContent = "Deleted models can be restored below.";
    $("model-action-note").textContent = "Create a new model or restore a deleted model.";
    $("test-results").innerHTML = '<h3>Held-out test results</h3><p class="hint">Train a model with a test split to evaluate it here.</p>';
    syncModelButtons();
    return;
  }
  info.innerHTML = `<strong>${escapeHTML(model.name)}</strong>${model.active ? `<span class="model-active"> · ${model.ready ? "Running on map" : "Selected · retraining required"}</span>` : ""}<br>${model.epoch ? `Epoch ${model.epoch} · ` : ""}${model.first_training_date ? `Trained ${escapeHTML(shortDate(model.first_training_date))} – ${escapeHTML(shortDate(model.last_training_date))}` : "Training dates unavailable"}<div class="model-split-counts"><span>${model.training_days} train days</span><span>${model.validation_days} validation days</span><span>${model.test_days} test days</span></div>${model.error ? `<p class="model-error">${escapeHTML(model.error)}</p>` : ""}`;
  $("model-action-note").textContent = model.active ? "This model currently powers the map. You can evaluate it without changing its weights." : "Selecting a model here does not change the map. Use for forecasts activates this model.";
  $("delete-model-note").textContent = !model.can_delete ? "This custom model is managed outside the library."
    : `${model.delete_scope === "run" ? "Moves this run's best and latest checkpoints, including test reports," : "Moves this checkpoint and its test report"} to Deleted. You can restore them below.${model.active ? " Forecasts will pause until you activate another model." : ""}`;
  syncModelButtons();
  const identifier = model.id;
  const park = state.park;
  const response = await api(`/api/test-report?model=${encodeURIComponent(identifier)}`);
  if (park !== state.park || identifier !== state.selectedModel) return;
  renderTestReport(response.report);
}

function renderDeletedModels(models) {
  $("deleted-model-count").textContent = models.length;
  const list = $("deleted-model-list");
  list.replaceChildren();
  if (!models.length) {
    const empty = document.createElement("p"); empty.className = "hint";
    empty.textContent = "No deleted models for this park."; list.append(empty);
  }
  for (const model of models) {
    const row = document.createElement("div"); row.className = "deleted-model";
    const description = document.createElement("div");
    const name = document.createElement("strong"); name.textContent = model.name;
    const detail = document.createElement("p"); detail.className = "hint";
    detail.textContent = `Deleted ${new Date(model.deleted_at).toLocaleString()} · ${model.scope === "run" ? "Entire model run" : "Checkpoint"}${model.reason ? ` · ${model.reason}` : ""}`;
    description.append(name, detail);
    const button = document.createElement("button");
    button.className = "secondary restore-model"; button.textContent = "Restore";
    button.dataset.available = String(model.can_restore);
    button.disabled = state.jobBusy || state.submitting || !model.can_restore;
    button.addEventListener("click", () => manage("restore", model.id));
    row.append(description, button); list.append(row);
  }
}

function renderTestReport(report) {
  const container = $("test-results");
  if (!report) {
    container.innerHTML = '<h3>Held-out test results</h3><p class="hint">No current test report for this model. Evaluate test split uses only the test dates saved during training, even after the park archive grows.</p>';
    return;
  }
  const number = value => typeof value === "number" && Number.isFinite(value) ? value.toFixed(2) : "N/A";
  const rows = Object.entries(report.horizons).map(([minutes, result]) => {
    const errors = result.wait_errors;
    return `<tr><td>${Number(minutes) < 60 ? `${minutes} min` : `${Number(minutes) / 60} h`}</td><td>${result.accuracy === null ? "N/A" : `${(result.accuracy * 100).toFixed(1)}%`}</td><td>${number(errors.absolute_minutes?.mean)}</td><td>${number(errors.mse_minutes_squared)}</td><td>${number(errors.rmse_minutes)}</td><td>${errors.count.toLocaleString()} / ${errors.actual_numeric_wait_count.toLocaleString()}</td></tr>`;
  }).join("");
  container.innerHTML = `<h3>Held-out test results</h3><p class="hint">${report.dates.length} saved test days · ${escapeHTML(shortDate(report.dates[0]))} – ${escapeHTML(shortDate(report.dates.at(-1)))}</p><div class="test-table-wrap"><table><thead><tr><th>Lead</th><th>Accuracy</th><th>MAE (min)</th><th>MSE (min²)</th><th>RMSE (min)</th><th>Numeric pairs</th></tr></thead><tbody>${rows}</tbody></table></div><p class="hint">Accuracy measures wait/status tokens. MAE, MSE (mean squared error), and RMSE use numeric wait pairs; N/A means no eligible forecasts. Weights stay unchanged during testing.</p><details><summary>Exact saved test dates (${report.dates.length})</summary>${report.dates.map(date => escapeHTML(date)).join(", ")}</details>`;
}

function showJobConsole(expanded = true) {
  $("studio-content").classList.toggle("log-expanded", expanded);
  $("expand-log").textContent = expanded ? "Show model controls" : "Expand log";
  $("expand-log").setAttribute("aria-pressed", String(expanded));
  $("studio-content").scrollTop = 0;
}

function renderCoverage() {
  const coverage = state.meta.coverage;
  $("coverage-count").textContent = `${coverage.cached_days.toLocaleString()} days`;
  $("coverage-first").textContent = coverage.first_date ? shortDate(coverage.first_date) : "No data yet";
  $("coverage-last").textContent = coverage.last_date ? shortDate(coverage.last_date) : "—";
  $("coverage-note").textContent = coverage.cached_days
    ? `${coverage.complete_days} complete · ${coverage.partial_days} partial · ${coverage.missing_days} uncached within this range`
    : "No observations saved for this park. Refresh data to get started.";
  $("coverage-note").title = `${coverage.observations.toLocaleString()} saved five-minute snapshots. Complete days are available for training; partial days can be explored and may support forecasts.`;
  $("header-range").textContent = coverage.cached_days ? `${coverage.cached_days} cached days · ${coverage.observations.toLocaleString()} snapshots` : "Local archive · no cached data";
  $("training-coverage").textContent = coverage.cached_days
    ? `${coverage.complete_days} complete days available across ${shortDate(coverage.first_date)} – ${shortDate(coverage.last_date)}. Training randomly assigns whole days to each split.`
    : "Refresh this park's data before training a model.";
  const strip = $("coverage-strip");
  strip.replaceChildren();
  for (const month of archiveMonths(state.meta.days)) {
    const button = document.createElement("button");
    const [year, number] = month.key.split("-").map(Number);
    const label = `${monthNames[number - 1]} ${year}: ${month.complete} complete, ${month.partial} partial days. Browse this month.`;
    button.className = "coverage-month";
    button.dataset.month = month.key;
    button.title = label;
    button.setAttribute("aria-label", label);
    for (const [name, count] of [["complete", month.complete], ["partial", month.partial]]) {
      const fill = document.createElement("span");
      fill.className = `${name}-fill`;
      fill.style.height = `${100 * count / month.calendarDays}%`;
      if (name === "partial") fill.style.bottom = `${100 * month.complete / month.calendarDays}%`;
      button.append(fill);
    }
    button.addEventListener("click", () => {
      setPanel("forecast");
      state.calendarMonth = month.key;
      $("calendar-browser").open = true;
      renderCalendar();
      $("calendar-browser").scrollIntoView({block: "nearest", behavior: "smooth"});
    });
    strip.append(button);
  }
}

function renderCalendar() {
  const calendar = $("calendar-days");
  calendar.replaceChildren();
  const enabled = state.dates.length > 0;
  for (const id of ["calendar-month", "calendar-year", "latest-day", "first-day"]) $(id).disabled = !enabled;
  $("previous-month").disabled = !enabled || state.calendarMonth <= state.dates[0].slice(0, 7);
  $("next-month").disabled = !enabled || state.calendarMonth >= state.dates.at(-1).slice(0, 7);
  const position = state.dates.indexOf(state.selectedDate);
  $("previous-day").disabled = position <= 0;
  $("next-day").disabled = position < 0 || position >= state.dates.length - 1;
  if (!enabled) {
    $("calendar-label").textContent = "No cached dates";
    $("month-count").textContent = "Refresh data to begin";
    $("calendar-year").replaceChildren();
    $("calendar-month").replaceChildren();
    return;
  }
  const [year, month] = state.calendarMonth.split("-").map(Number);
  $("calendar-label").textContent = `${monthNames[month - 1]} ${year}`;
  const yearSelect = $("calendar-year"); yearSelect.replaceChildren();
  for (let value = Number(state.dates[0].slice(0, 4)); value <= Number(state.dates.at(-1).slice(0, 4)); value++) {
    const option = document.createElement("option"); option.value = value; option.textContent = value; yearSelect.append(option);
  }
  yearSelect.value = String(year);
  const monthSelect = $("calendar-month"); monthSelect.replaceChildren();
  monthNames.forEach((name, index) => {
    const option = document.createElement("option"); option.value = index + 1; option.textContent = name;
    const key = `${year}-${String(index + 1).padStart(2, "0")}`;
    option.disabled = key < state.dates[0].slice(0, 7) || key > state.dates.at(-1).slice(0, 7);
    monthSelect.append(option);
  });
  monthSelect.value = String(month);
  const offset = (new Date(Date.UTC(year, month - 1, 1)).getUTCDay() + 6) % 7;
  for (let index = 0; index < offset; index++) calendar.append(document.createElement("span"));
  const count = new Date(Date.UTC(year, month, 0)).getUTCDate();
  for (let value = 1; value <= count; value++) {
    const key = `${state.calendarMonth}-${String(value).padStart(2, "0")}`;
    const day = state.daysByDate.get(key);
    const button = document.createElement("button");
    button.textContent = value;
    button.disabled = !day;
    button.className = day && !day.complete ? "partial" : "";
    button.setAttribute("aria-pressed", String(state.selectedDate === key));
    button.setAttribute("aria-label", `${shortDate(key)}${day ? `, ${day.complete ? "complete" : "partial"} data, ${day.observations} snapshots` : ", no cached data"}`);
    button.title = day ? `${day.observations}/${day.expected} snapshots${day.reason ? ` · ${day.reason}` : ""}` : "No data cached for this date";
    button.addEventListener("click", () => chooseDay(key));
    calendar.append(button);
  }
  const inMonth = state.meta.days.filter(day => day.date.startsWith(state.calendarMonth));
  $("month-count").textContent = `${inMonth.length} of ${count} days cached`;
  for (const button of document.querySelectorAll(".coverage-month")) {
    button.classList.toggle("active", button.dataset.month === state.calendarMonth);
    button.setAttribute("aria-pressed", String(button.dataset.month === state.calendarMonth));
  }
}

async function chooseDay(key) {
  if (!state.daysByDate.has(key)) {
    $("day").value = state.selectedDate || "";
    message("No data is cached for that date. Choose a marked day in the calendar.", true);
    return;
  }
  state.selectedDate = key;
  state.calendarMonth = key.slice(0, 7);
  $("day").value = key;
  $("calendar-browser").open = false;
  renderCalendar();
  await loadDay(key);
}

function changeMonth(offset) {
  if (!state.calendarMonth || !state.dates.length) return;
  const value = monthNumber(state.calendarMonth) + offset;
  const first = monthNumber(state.dates[0]), last = monthNumber(state.dates.at(-1));
  state.calendarMonth = monthKey(Math.max(first, Math.min(last, value)));
  renderCalendar();
}

function updatePresets() {
  for (const button of document.querySelectorAll("[data-start]")) {
    let valid = false;
    if (state.day && state.meta.checkpoint_available) {
      try { valid = resolveClock(button.dataset.start) > state.day.timestamps[0]; } catch { /* Outside this day's schedule. */ }
    }
    button.disabled = !valid;
    button.classList.toggle("active", button.dataset.start === $("start").value);
  }
}

async function api(path, options = {}) {
  const separator = path.includes("?") ? "&" : "?";
  const response = await fetch(`${path}${separator}park=${encodeURIComponent(state.park || "disneyland")}`, options);
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
  // Global aerial coverage for both Anaheim and Orlando.
  state.layers.aerial = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
    maxNativeZoom: 19, maxZoom: 21,
    attribution: 'Imagery &copy; <a href="https://www.esri.com/">Esri</a>, Maxar, Earthstar Geographics, and the GIS User Community',
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

function restoreRecordedFrame() {
  if (!state.day) return;
  state.cutoff = Infinity;
  state.predicted.clear();
  const viewed = state.day.timestamps[state.index];
  if (!state.observed.has(viewed) && state.observed.size) {
    const recorded = [...state.observed.keys()].sort((a, b) => a - b);
    const tick = recorded.filter(value => value <= viewed).at(-1) ?? recorded[0];
    state.index = Math.max(0, state.day.timestamps.indexOf(tick));
  }
  renderTimeline();
  setIndex(state.index);
}

function renderModelStatus() {
  const status = state.meta.checkpoint_status;
  const ready = state.meta.checkpoint_available;
  const incompatible = status && ["needs_training", "invalid"].includes(status.state);
  $("forecast-model-badge").textContent = incompatible ? "Retraining required" : !ready ? "No model loaded"
    : state.meta.checkpoint.includes("saved") ? "Saved model" : state.meta.checkpoint.includes("latest") ? "Latest model" : "Best model";
  $("model-warning").hidden = !incompatible;
  $("model-warning").textContent = incompatible ? status.message : "";
  $("review-model").hidden = ready;
  $("active-model-name").textContent = status?.run_name || (ready ? "Previous best model" : incompatible ? "Retraining required" : "No saved model is running");
  if (ready && status?.run_name) { $("forecast-model-badge").textContent = status.run_name; $("forecast-model-badge").title = status.run_name; }
  const training = status?.first_training_date && status?.last_training_date
    ? `Trained on ${status.training_days} days (${shortDate(status.first_training_date)} – ${shortDate(status.last_training_date)}). ` : "";
  $("model-status").textContent = training + (status?.message || (ready ? `Active: ${state.meta.checkpoint}` : "No model trained yet."));
}

async function loadDay(key) {
  const generation = ++state.generation;
  stopPlayback();
  for (const id of ["start", "view-time", "timeline", "play", "back", "forward"]) $(id).disabled = true;
  state.day = null;
  state.observed.clear();
  state.predicted.clear();
  for (const marker of state.markers.values()) if (state.map) state.map.removeLayer(marker);
  state.markers.clear();
  $("ride-list").replaceChildren();
  $("ride-count").textContent = "0";
  $("clock").textContent = "—";
  $("source").textContent = "Loading operating day";
  $("source-note").textContent = dateFormatter.format(dateObject(key));
  $("map-selected-date").textContent = dateFormatter.format(dateObject(key));
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
    const zone = new Intl.DateTimeFormat("en-US", {timeZone: state.meta.timezone, timeZoneName: "short"})
      .formatToParts(new Date(day.timestamps[0] * 1000)).find(part => part.type === "timeZoneName").value;
    $("zone").textContent = zone;
    const recorded = day.snapshots.filter(frame => frame.source === "observed").length;
    const interpolated = day.snapshots.length - recorded;
    const summary = state.daysByDate.get(key);
    $("day-note").classList.toggle("partial", Boolean(summary && !summary.complete));
    $("day-note").textContent = `${recorded}/${day.expected_timestamps.length} recorded snapshots${interpolated ? ` · ${interpolated} interpolated` : ""}${summary && !summary.complete ? " · Partial day" : " · Complete day"}`;
    $("day-note").title = day.reason || "All scheduled five-minute observations are cached.";
    $("map-selected-date").textContent = dateFormatter.format(dateObject(key));
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
      message(state.meta.checkpoint_status?.message || "No trained model found. Open Data & models to train one. Recorded waits remain available.");
      $("forecast-note").textContent = state.meta.checkpoint_status?.state === "needs_training"
        ? "Retrain this park's model to enable forecasts. Refreshing data does not train the model."
        : "Train or load a compatible model to enable forecasts.";
    } else message(state.meta.location_warning);
    state.cutoff = Infinity;
    const selectedTime = day.timestamps.find(tick => clockInput(tick) === $("start").value);
    setIndex(Math.max(0, day.timestamps.indexOf(selectedTime ?? day.latest_observation)));
    restoreRecordedFrame();
    updatePresets();
    $("generate").disabled = !state.meta.checkpoint_available;
    $("generate-label").textContent = "Generate forecast";
    renderTimeline();
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
  $("forecast-note").textContent = `Generating predictions from ${clock(cutoff)}…`;
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
        const method = result.forecast_method === "direct_probabilities" ? "Stable availability forecast" : `${result.samples} paths`;
        $("forecast-note").textContent = `History through ${clock(Date.parse(result.as_of) / 1000)} · ${method}. Dashed pins are predictions.`;
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
    restoreRecordedFrame();
    message(error.message, true);
    $("forecast-note").textContent = error.message;
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

function predictionError(ride, frame) {
  if (frame.source !== "predicted" || frame.pending) return null;
  const recorded = state.observed.get(frame.timestamp);
  if (recorded?.source !== "observed") return null;
  const actual = recorded.rides.find(row => row.id === ride.id);
  const known = row => row && (
    (row.status === "OPERATING" && (Number.isFinite(row.wait) || row.wait === "300+"))
    || ["DOWN", "CLOSED", "REFURBISHMENT"].includes(row.status)
  );
  if (!known(actual) || !known(ride)) return null;
  const comparison = `Predicted ${displayWait(ride, true)}; recorded ${displayWait(actual, true)}.`;
  if (Number.isFinite(ride.wait) && Number.isFinite(actual.wait)) {
    const error = ride.wait - actual.wait;
    const signed = `${error > 0 ? "+" : ""}${error}`;
    return {label: `${signed} min`, compact: `${signed}m`,
      severity: Math.abs(error) <= 5 ? "small" : Math.abs(error) <= 15 ? "medium" : "large",
      title: `${comparison} Error = predicted − recorded; positive means overestimated.`};
  }
  const match = ride.status === actual.status && ride.wait === actual.wait;
  return {label: match ? "Match" : "Mismatch", compact: match ? "Match" : "Mismatch",
    severity: match ? "small" : "large",
    title: `${comparison} ${match ? "Matching wait/status token." : "Wait/status token mismatch; no exact minute error is available."}`};
}

function errorBadge(ride, frame, compact = false) {
  const error = predictionError(ride, frame);
  if (!error) return "";
  return `<span class="prediction-error ${error.severity}" title="${escapeHTML(error.title)}" aria-label="${escapeHTML(`Prediction error: ${error.label}. ${error.title}`)}">${escapeHTML(compact ? error.compact : error.label)}</span>`;
}

function render() {
  if (!state.day) return;
  const tick = state.day.timestamps[state.index];
  const frame = currentFrame();
  $("clock").textContent = clock(tick);
  $("timeline").setAttribute("aria-valuetext", `${clock(tick)}, ${frame.pending ? "prediction unavailable" : sourceNames[frame.source]}`);
  const localDay = new Intl.DateTimeFormat("en-CA", {timeZone: state.meta.timezone, year: "numeric", month: "2-digit", day: "2-digit"}).format(new Date(tick * 1000));
  $("day-offset").textContent = localDay !== state.day.date ? "+1 day" : "";
  $("source").className = `source ${frame.source}`;
  $("source").textContent = frame.pending ? (state.loading ? "Preparing prediction" : "Prediction unavailable") : sourceNames[frame.source];
  $("source-note").textContent = frame.pending ? (state.forecastError ? "Adjust the prediction start to resolve the error." : "Generate a forecast to explore this time.")
    : frame.source === "predicted" ? (frame.forecast_method === "direct_probabilities"
      ? "Stable availability · median wait when operating" : `${state.meta.samples} paths · median wait when operating`)
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
      : frame.source === "predicted" ? `Forecast · ${Math.round((ride.down_probability || 0) * 100)}% down · ${Math.round((ride.closed_probability || 0) * 100)}% closed`
      : ride.status === "MISSING" ? "Missing observation"
      : frame.source === "interpolated" ? "Interpolated wait" : "Recorded wait";
    label.append(name, subtitle);
    const pill = document.createElement("span");
    pill.className = `wait-pill ${frame.source}${ride.wait === null ? " inactive" : ""}`;
    pill.textContent = displayWait(ride);
    if (ride.wait !== null) {
      const unit = document.createElement("small"); unit.textContent = "min"; pill.append(unit);
    }
    const values = document.createElement("span");
    values.className = "ride-values";
    values.append(pill);
    values.insertAdjacentHTML("beforeend", errorBadge(ride, frame));
    button.append(label, values);
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
    const html = `<div class="map-pin-group"><div class="map-pin ${frame.source}${inactive ? " inactive" : ""}${state.selected === ride.id ? " selected" : ""}">${escapeHTML(text)}</div>${errorBadge(ride, frame, true)}</div>`;
    const icon = L.divIcon({className: "wait-marker", html, iconSize: [43, 32], iconAnchor: [21, 35], tooltipAnchor: [0, -35]});
    let marker = state.markers.get(ride.id);
    if (!marker) {
      marker = L.marker([point.latitude, point.longitude], {icon, title: ride.name, alt: ride.name, riseOnHover: true})
        .on("click", () => selectRide(ride.id)).addTo(state.map);
      marker.bindTooltip("", {direction: "top"});
      state.markers.set(ride.id, marker);
    } else marker.setIcon(icon);
    const error = predictionError(ride, frame);
    marker.setTooltipContent(`${escapeHTML(ride.name)}<br>${escapeHTML(text)}${!inactive ? " min" : ""} · ${frame.pending ? "Prediction unavailable" : sourceNames[frame.source]}${error ? `<br>${escapeHTML(error.title)}` : ""}`);
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
  $("detail-wait").insertAdjacentHTML("beforeend", errorBadge(ride, frame));
  $("detail-meta").textContent = frame.pending ? "Generate a forecast for this time."
    : frame.source === "predicted" ? `${ride.wait !== null ? `${ride.p10 ?? "—"}–${ride.p90 ?? "—"} min p10–p90 when operating · ` : ""}${Math.round((ride.down_probability || 0) * 100)}% down · ${Math.round((ride.closed_probability || 0) * 100)}% closed`
    : frame.source === "interpolated" ? "Linearly interpolated from surrounding observations; rounded to five minutes."
    : "Waits are saved in five-minute buckets. Gaps in the chart indicate unavailable waits.";
  const error = predictionError(ride, frame);
  if (error) $("detail-meta").textContent += ` ${error.title}`;
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
$("day").addEventListener("change", () => chooseDay($("day").value));
$("latest-day").addEventListener("click", () => chooseDay(state.dates.at(-1)));
$("first-day").addEventListener("click", () => chooseDay(state.dates[0]));
$("previous-day").addEventListener("click", () => chooseDay(state.dates[state.dates.indexOf(state.selectedDate) - 1]));
$("next-day").addEventListener("click", () => chooseDay(state.dates[state.dates.indexOf(state.selectedDate) + 1]));
$("previous-month").addEventListener("click", () => changeMonth(-1));
$("next-month").addEventListener("click", () => changeMonth(1));
$("calendar-month").addEventListener("change", () => {
  state.calendarMonth = `${$("calendar-year").value}-${String($("calendar-month").value).padStart(2, "0")}`;
  renderCalendar();
});
$("calendar-year").addEventListener("change", () => {
  state.calendarMonth = `${$("calendar-year").value}-${state.calendarMonth.slice(5)}`;
  changeMonth(0);
});
for (const panel of ["forecast", "manage"]) {
  $(`${panel}-tab`).addEventListener("click", () => setPanel(panel));
  $(`${panel}-tab`).addEventListener("keydown", event => {
    if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      setPanel(event.key === "Home" ? "forecast" : event.key === "End" ? "manage" : panel === "forecast" ? "manage" : "forecast", true);
    }
  });
}

function startChanged() {
  if (!state.day) return;
  ++state.generation;
  state.loading = false;
  restoreRecordedFrame();
  $("generate").disabled = !state.meta.checkpoint_available;
  $("generate-label").textContent = "Generate forecast";
  $("forecast-note").classList.remove("error");
  $("forecast-note").textContent = "Uses saved observations before the start time.";
  updatePresets();
  renderTimeline();
  render();
}
for (const button of document.querySelectorAll("[data-start]")) {
  button.addEventListener("click", () => { $("start").value = button.dataset.start; startChanged(); });
}
$("generate").addEventListener("click", generate);
$("start").addEventListener("change", startChanged);
$("review-model").addEventListener("click", () => { setPanel("manage"); $("training-setup").open = true; $("run-name").focus(); });
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
$("recenter").addEventListener("click", () => { if (state.map && state.meta) state.map.flyTo([state.meta.park.latitude, state.meta.park.longitude], state.meta.park.zoom, {duration: .5}); });
$("close-detail").addEventListener("click", () => { state.selected = null; $("detail").hidden = true; render(); });
async function reloadPark() {
  const selectedPark = state.park;
  ++state.generation;
  stopPlayback();
  state.day = null;
  state.meta = null;
  state.dates = [];
  state.daysByDate.clear();
  state.selectedDate = null;
  state.calendarMonth = null;
  renderCalendar();
  state.predicted.clear();
  state.observed.clear();
  state.selected = null;
  state.cutoff = Infinity;
  $("detail").hidden = true;
  $("coverage-count").textContent = "…";
  $("coverage-first").textContent = "Loading archive…";
  $("coverage-last").textContent = "—";
  $("coverage-note").textContent = "Reading this park's cached date range.";
  $("coverage-strip").replaceChildren();
  $("header-range").textContent = "Loading park coverage…";
  $("forecast-model-badge").textContent = "Loading model…";
  for (const marker of state.markers.values()) if (state.map) state.map.removeLayer(marker);
  state.markers.clear();
  for (const id of ["day", "start", "view-time", "timeline", "play", "back", "forward", "generate"]) $(id).disabled = true;
  $("ride-list").replaceChildren();
  $("ride-count").textContent = "0";
  $("clock").textContent = "—";
  $("source").textContent = "No day selected";
  $("source-note").textContent = "Select a cached operating day to explore waits.";
  $("forecast-note").textContent = "Uses saved observations before the start time.";
  $("timeline-labels").replaceChildren();
  $("timeline").style.removeProperty("--track");
  $("boundary").hidden = true;
  $("zone").textContent = "";
  $("day").value = "";
  $("map-selected-date").textContent = "Choose an operating date";
  for (const button of document.querySelectorAll("[data-start]")) button.disabled = true;
  message("Loading this park's archive…");
  try {
    const meta = await api("/api/meta");
    if (selectedPark !== state.park) return;
    state.meta = meta;
    clockFormatter = new Intl.DateTimeFormat("en-US", {timeZone: state.meta.timezone, hour: "numeric", minute: "2-digit"});
    inputFormatter = new Intl.DateTimeFormat("en-GB", {timeZone: state.meta.timezone, hour: "2-digit", minute: "2-digit", hourCycle: "h23"});
    document.title = `OmniQueue · ${state.meta.park.name}`;
    $("park-name").textContent = state.meta.park.name;
    $("map-park-name").textContent = state.meta.park.name;
    $("local-time-label").textContent = `All times in ${state.meta.park.name} local time `;
    if (state.map) state.map.setView([state.meta.park.latitude, state.meta.park.longitude], state.meta.park.zoom);
    state.dates = state.meta.days.map(day => day.date).sort();
    state.daysByDate = new Map(state.meta.days.map(day => [day.date, day]));
    renderCoverage();
    renderModelStatus();
    if (state.loadedPark !== selectedPark) {
      state.loadedPark = selectedPark;
      state.models = [];
      state.modelPreview = null;
      state.selectedModel = null;
      state.jobBusy = false;
      state.handledJob = null;
      $("run-name").value = "";
      $("copy-name").value = "";
      $("studio-content").classList.remove("log-expanded");
    }
    $("studio-title").textContent = `Models for ${state.meta.park.name}`;
    if (!$("model-workspace").hidden) await loadModels();
    const select = $("day");
    if (!state.meta.days.length) {
      message("No cached data for this park yet. Select Refresh data to download history and recent Firebase observations.");
      $("day-note").textContent = "No cached days";
      return;
    }
    select.min = state.dates[0];
    select.max = state.dates.at(-1);
    select.disabled = false;
    $("start").value = state.meta.default_start || "12:00";
    await chooseDay(state.meta.default_day);
  } catch (error) {
    message(error.message, true);
    $("source").textContent = "Archive unavailable";
    $("day-note").textContent = error.message;
  }
}
let taskTimer = null;
function renderStopTraining(task) {
  state.jobId = task.id || null;
  $("stop-training").hidden = task.action !== "train" || !["queued", "running", "cancelling"].includes(task.state);
  $("stop-training").disabled = task.state === "cancelling" || state.stopping;
  $("stop-training").textContent = task.state === "cancelling" ? "Stopping & discarding..." : "Stop & discard model";
}

function renderAllLive(task) {
  state.liveAllBusy = ["queued", "running"].includes(task?.state);
  $("live-all-data").disabled = state.liveAllSubmitting || state.liveAllBusy;
  if (!task) return false;
  $("live-all-status").textContent = task.state === "queued" ? "Queued for live refresh."
    : task.state === "running" ? `Refreshing live data. ${task.current}/${task.total} parks processed.`
    : task.state === "complete" ? `Live refresh finished for all ${task.total} parks.`
    : `${task.current}/${task.total} parks processed. ${task.error}`;
  if (state.liveAllBusy || state.liveAllHandled === task.id) return false;
  state.liveAllHandled = task.id;
  return true;
}

async function refreshAllLive() {
  if (state.liveAllSubmitting || state.liveAllBusy) return;
  state.liveAllSubmitting = true;
  $("live-all-data").disabled = true;
  $("live-all-status").textContent = "Starting live refresh…";
  try {
    await api("/api/action", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({action: "live-all"})});
    await pollTask();
  } catch (error) {
    $("live-all-status").textContent = error.message;
  } finally {
    state.liveAllSubmitting = false;
    $("live-all-data").disabled = state.liveAllBusy;
  }
}

async function pollTask() {
  clearTimeout(taskTimer);
  const park = state.park;
  try {
    const task = await api("/api/task");
    if (park !== state.park) return;
    const refreshFinished = renderAllLive(task.live_all);
    const busy = ["queued", "running", "cancelling"].includes(task.state);
    state.jobBusy = busy;
    renderStopTraining(task);
    $("task-progress").hidden = !busy && !task.total;
    $("job-dot").hidden = !busy;
    $("task-state").textContent = task.state[0].toUpperCase() + task.state.slice(1);
    $("job-name").textContent = task.name ? `${task.action === "train" ? "Training" : task.action === "test" ? "Testing" : "Refreshing"}: ${task.name}` : "Training & evaluation log";
    $("task-status").textContent = task.error || (task.state === "cancelled" ? "Training stopped. The unfinished model was discarded. The previously selected model is unchanged." : task.state === "cancelling" ? "Stopping the training process and removing this run's unfinished model files..." : task.state === "idle" ? "Ready when you are." : task.state === "complete" ? (task.action === "train" ? "Training finished. This named model is saved in the model library." : task.action === "test" ? "Evaluation finished. Results are saved with this model." : "Data refresh finished.") : task.state === "queued" ? "Queued behind another park's job. Your saved progress is safe." : "Working — output appears below as each step finishes.");
    if (task.total) { $("task-progress").max = task.total; $("task-progress").value = task.current; }
    else $("task-progress").removeAttribute("value");
    const elapsed = task.elapsed_seconds ? ` · ${Math.floor(task.elapsed_seconds / 60)}m ${task.elapsed_seconds % 60}s` : "";
    $("job-progress-text").textContent = task.total ? `${task.action === "test" ? "Test days" : "Epochs"} ${task.current}/${task.total}${elapsed}` : task.state === "idle" ? "Start training or evaluate a saved model." : `${task.state}${elapsed}`;
    const logText = task.log.length ? task.log.join("\n") : busy ? "Starting job..." : "Your training progress and test metrics will appear here.";
    const logPosition = $("task-log").scrollTop;
    if ($("task-log").textContent !== logText) $("task-log").textContent = logText;
    $("task-log").scrollTop = $("follow-log").checked ? $("task-log").scrollHeight : logPosition;
    $("download-log").disabled = !task.id;
    syncModelButtons();
    let reloaded = false;
    if (task.id && !busy && state.handledJob !== task.id) {
      state.handledJob = task.id;
      if (task.state === "cancelled") state.selectedModel = null;
      else if (task.model_id) state.selectedModel = task.model_id;
      if (["complete", "cancelled"].includes(task.state)) {
        await reloadPark();
        reloaded = true;
        if (task.action === "test") {
          showJobConsole(false);
          $("training-setup").open = false;
          $("model-library").open = true;
          $("test-results").scrollIntoView({block: "nearest", behavior: "smooth"});
        }
      } else {
        if (task.action === "train") await loadModels();
        message(task.error, true);
      }
    }
    if (refreshFinished && !reloaded) await reloadPark();
  } catch (error) { $("task-status").textContent = error.message; }
  taskTimer = setTimeout(pollTask, 1500);
}

async function manage(action, deletedId = null) {
  if (state.jobBusy || state.submitting) return;
  if (action === "refresh" || action === "rebuild") {
    const prompt = action === "rebuild"
      ? "Re-fetch history? This downloads historical data again, replacing cached historical downloads, and refreshes recent Firebase observations. This can take a while. Continue?"
      : "Refresh all data? This fetches historical data as well as recent Firebase observations. Historical downloads can take a while. Continue?";
    if (!window.confirm(prompt)) return;
  }
  const park = state.park;
  state.submitting = true;
  syncModelButtons();
  try {
    const payload = {park, action, deleted_id: deletedId, epochs: Number($("epochs").value), model: state.selectedModel,
      name: action === "save" ? $("copy-name").value.trim() : $("run-name").value.trim(),
      split: splitValues(), use_after_train: $("use-after-train").checked};
    const result = await api("/api/action", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
    if (park !== state.park) return;
    if (result.model_id) state.selectedModel = result.model_id;
    if (result.state === "complete") {
      if (action === "load") { await reloadPark(); setPanel("forecast"); }
      else if (action === "delete" || action === "restore") {
        if (action === "delete") state.selectedModel = null;
        await reloadPark();
        $("model-action-note").textContent = result.message;
      }
      else { await loadModels(result.model_id); $("model-action-note").textContent = result.message; $("copy-name").value = ""; }
      message(result.message || "Saved model updated.");
    } else {
      state.jobBusy = true;
      $("job-name").textContent = `${action === "train" ? "Training" : action === "test" ? "Testing" : "Refreshing"}: ${payload.name}`;
      showJobConsole(true);
    }
    await pollTask();
  } catch (error) {
    $("task-status").textContent = error.message;
    $("model-action-note").textContent = error.message;
    message(error.message, true);
  } finally {
    if (park === state.park) { state.submitting = false; syncModelButtons(); }
  }
}
$("park").addEventListener("change", async () => {
  state.park = $("park").value;
  state.jobBusy = false;
  state.submitting = false;
  state.jobId = null;
  state.stopping = false;
  $("stop-training").hidden = true;
  await reloadPark();
  await pollTask();
});
for (const [id, action] of [["refresh-data", "refresh"], ["rebuild-data", "rebuild"], ["live-data", "live"], ["train-model", "train"], ["test-model", "test"], ["load-model", "load"], ["save-model", "save"], ["delete-model", "delete"]]) {
  $(id).addEventListener("click", () => manage(action));
}
$("live-all-data").addEventListener("click", refreshAllLive);
$("model-slot").addEventListener("change", () => {
  state.selectedModel = $("model-slot").value;
  renderSavedModel().catch(error => { $("model-action-note").textContent = error.message; });
});
for (const id of ["run-name", "copy-name"]) $(id).addEventListener("input", syncModelButtons);
let splitTimer = null;
for (const id of ["split-train", "split-validation", "split-test"]) $(id).addEventListener("input", () => {
  clearTimeout(splitTimer);
  renderSplitPreview({ready: false, error: "Updating day split..."});
  splitTimer = setTimeout(() => loadModels().catch(error => renderSplitPreview({ready: false, error: error.message})), 300);
});
$("expand-log").addEventListener("click", () => showJobConsole(!$("studio-content").classList.contains("log-expanded")));
$("focus-training").addEventListener("click", () => {
  showJobConsole(false); $("training-setup").open = true; $("studio-content").scrollTop = 0; $("run-name").focus();
});
$("focus-library").addEventListener("click", () => {
  showJobConsole(false); $("model-library").open = true; $("model-library").scrollIntoView({block:"nearest", behavior:"smooth"});
});
$("stop-training").addEventListener("click", async () => {
  if (!state.jobId || state.stopping) return;
  const park = state.park;
  state.stopping = true;
  $("stop-training").disabled = true;
  try {
    await api("/api/action", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({park, action: "cancel", job_id: state.jobId})});
    if (park === state.park) await pollTask();
  } catch (error) { $("task-status").textContent = error.message; }
  finally { state.stopping = false; }
});
$("download-log").addEventListener("click", async () => {
  try {
    const response = await fetch(`/api/task-log?park=${encodeURIComponent(state.park)}`);
    if (!response.ok) throw new Error("Could not download the job log.");
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a"); link.href = url; link.download = `${state.park}-model-job.log`; link.click(); URL.revokeObjectURL(url);
  } catch (error) { $("task-status").textContent = error.message; }
});

async function init() {
  initMap();
  try {
    const result = await api("/api/parks");
    $("park").replaceChildren();
    for (const park of result.parks) {
      const option = document.createElement("option");
      option.value = park.slug; option.textContent = park.name; $("park").append(option);
    }
    state.park = result.default;
    $("park").value = state.park;
    await reloadPark();
    await pollTask();
  } catch (error) { message(error.message, true); }
}
init();
