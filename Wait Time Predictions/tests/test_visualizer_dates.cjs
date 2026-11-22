// Calendar behavior across years, gaps, partial days and empty park archives.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function calendar(days) {
  const elements = new Map();
  function element() {
    const classes = new Set();
    return {children: [], value: "", disabled: false, attributes: {}, dataset: {}, handlers: {},
      style: {setProperty() {}, removeProperty() {}},
      classList: {toggle(name, value) { value ? classes.add(name) : classes.delete(name); }, remove(name) { classes.delete(name); }, add(name) { classes.add(name); }, contains(name) { return classes.has(name); }},
      append(...items) { this.children.push(...items); }, replaceChildren() { this.children = []; },
      setAttribute(name, value) { this.attributes[name] = value; },
      addEventListener(name, callback) { this.handlers[name] = callback; }, focus() {}, scrollIntoView() {},
    };
  }
  const document = {getElementById(id) {
    if (!elements.has(id)) elements.set(id, element());
    return elements.get(id);
  }, createElement: element, querySelectorAll(selector) {
    if (selector === ".restore-model") return document.getElementById("deleted-model-list").children.flatMap(row => row.children.filter(child => child.className === "secondary restore-model"));
    return selector === ".coverage-month" ? document.getElementById("coverage-strip").children : [];
  }};
  const context = vm.createContext({Intl, Date, Map, Set, document});
  const source = fs.readFileSync(path.join(__dirname, "../ml/visualizer/app.js"), "utf8");
  vm.runInContext(source.replace(/\ninit\(\);\s*$/, "\n"), context);
  const ui = vm.runInContext("({state, archiveMonths, renderCalendar, chooseDay, changeMonth, setPanel, restoreRecordedFrame, renderModelStatus, syncModelButtons, renderSplitPreview, renderTestReport, renderDeletedModels, showJobConsole, renderStopTraining})", context);
  ui.state.meta = {days};
  ui.state.dates = days.map(day => day.date).sort();
  ui.state.daysByDate = new Map(days.map(day => [day.date, day]));
  ui.state.selectedDate = ui.state.dates.at(-1);
  ui.state.calendarMonth = ui.state.dates.at(-1)?.slice(0, 7) ?? null;
  vm.runInContext("var loadedDate = null; loadDay = async key => { loadedDate = key; };", context);
  return {...ui, elements, context, get: document.getElementById};
}

const day = (date, complete = true) => ({date, complete, observations: 100, expected: 100, reason: ""});

test("a year-plus archive keeps empty months visible and leap years correct", () => {
  const ui = calendar([]);
  const months = ui.archiveMonths([day("2024-02-29"), day("2025-03-02", false)]);
  assert.equal(months.length, 14);
  assert.equal(months[0].calendarDays, 29);
  assert.equal(months[1].total, 0);
  assert.equal(months.at(-1).partial, 1);
});

test("calendar marks complete and partial dates and disables uncached dates", () => {
  const ui = calendar([day("2025-11-01"), day("2026-11-01"), day("2026-11-03", false)]);
  ui.renderCalendar();
  const buttons = ui.get("calendar-days").children.filter(node => node.handlers.click);
  assert.equal(buttons.length, 30);
  assert.equal(buttons[0].disabled, false);
  assert.equal(buttons[1].disabled, true);
  assert.equal(buttons[2].className, "partial");
  assert.equal(buttons[2].attributes["aria-pressed"], "true");
  assert.equal(ui.get("previous-day").disabled, false);
  assert.equal(ui.get("next-day").disabled, true);
});

test("previous cached date skips gaps and can cross a year boundary", async () => {
  const ui = calendar([day("2025-12-29"), day("2026-01-03")]);
  ui.renderCalendar();
  await ui.get("previous-day").handlers.click();
  assert.equal(ui.state.selectedDate, "2025-12-29");
  assert.equal(ui.state.calendarMonth, "2025-12");
  assert.equal(ui.get("day").value, "2025-12-29");
  assert.equal(vm.runInContext("loadedDate", ui.context), "2025-12-29");
  assert.equal(ui.get("previous-day").disabled, true);
});

test("typed dates without cached data never issue a day request", async () => {
  const ui = calendar([day("2025-12-29"), day("2026-01-03")]);
  await ui.chooseDay("2026-01-01");
  assert.equal(vm.runInContext("loadedDate", ui.context), null);
  assert.equal(ui.get("day").value, "2026-01-03");
  assert.match(ui.get("message").textContent, /No data is cached/);
});

test("year jumps clamp to archive bounds and never manufacture unavailable years", () => {
  const ui = calendar([day("2025-10-18"), day("2026-11-20")]);
  ui.renderCalendar();
  ui.get("calendar-year").value = "2025";
  ui.get("calendar-year").handlers.change();
  assert.equal(ui.state.calendarMonth, "2025-11");
  ui.changeMonth(-100);
  assert.equal(ui.state.calendarMonth, "2025-10");
  ui.changeMonth(100);
  assert.equal(ui.state.calendarMonth, "2026-11");
  assert.equal(ui.get("calendar-year").children.length, 2);
});

test("switching to an empty park disables date navigation and clears the calendar", () => {
  const ui = calendar([]);
  ui.renderCalendar();
  for (const id of ["previous-day", "next-day", "previous-month", "next-month", "calendar-year", "calendar-month", "first-day", "latest-day"]) {
    assert.equal(ui.get(id).disabled, true, id);
  }
  assert.equal(ui.get("calendar-days").children.length, 0);
});

test("workspace tabs expose only the selected panel and support keyboard switching", () => {
  const ui = calendar([]);
  ui.setPanel("manage");
  assert.equal(ui.get("forecast-panel").hidden, true);
  assert.equal(ui.get("manage-panel").hidden, false);
  ui.get("manage-tab").handlers.keydown({key: "ArrowLeft", preventDefault() {}});
  assert.equal(ui.get("forecast-panel").hidden, false);
  assert.equal(ui.get("manage-tab").attributes["aria-selected"], "false");
});

test("a failed forecast returns a future view to the latest actual observation", () => {
  const ui = calendar([]);
  vm.runInContext("render = () => {}; renderTimeline = () => {};", ui.context);
  ui.state.day = {timestamps: [100, 400, 700, 1000]};
  ui.state.index = 3;
  ui.state.cutoff = 400;
  for (const tick of [100, 400, 700]) ui.state.observed.set(tick, {timestamp: tick});
  ui.state.predicted.set(1000, {timestamp: 1000});
  ui.restoreRecordedFrame();
  assert.equal(ui.state.index, 2);
  assert.equal(ui.state.cutoff, Infinity);
  assert.equal(ui.state.predicted.size, 0);
  assert.equal(ui.state.observed.size, 3);
});

test("incompatible saved models show retraining instructions instead of a best-model badge", () => {
  const ui = calendar([]);
  ui.state.meta = {checkpoint_available: false, checkpoint: "waits.pt", checkpoint_status: {
    state: "needs_training", message: "Open Data & models and choose Train & save best.",
    training_days: 21, first_training_date: "2026-10-18", last_training_date: "2026-11-07",
  }};
  ui.renderModelStatus();
  assert.equal(ui.get("forecast-model-badge").textContent, "Retraining required");
  assert.equal(ui.get("review-model").hidden, false);
  assert.match(ui.get("model-status").textContent, /Trained on 21 days/);
  assert.match(ui.get("model-warning").textContent, /Train & save best/);
});

test("model controls require a name, a valid split, and the selected model's test partition", () => {
  const ui = calendar([]);
  ui.state.models = [{id: "model-a", ready: true, test_ready: false, error: "No test split", test_days: 0}];
  ui.state.selectedModel = "model-a";
  ui.state.modelPreview = {ready: true};
  ui.get("run-name").value = "My model";
  ui.get("copy-name").value = "A copy";
  ui.syncModelButtons();
  assert.equal(ui.get("train-model").disabled, false);
  assert.equal(ui.get("load-model").disabled, false);
  assert.equal(ui.get("test-model").disabled, true);
  ui.renderSplitPreview({ready: false, error: "Percentages must total 100%"});
  assert.equal(ui.get("train-model").disabled, true);
  ui.state.jobBusy = true;
  ui.syncModelButtons();
  assert.equal(ui.get("load-model").disabled, true);
  assert.equal(ui.get("save-model").disabled, true);
});

test("test reports display saved dates and unavailable horizons without inventing errors", () => {
  const ui = calendar([]);
  ui.renderTestReport({dates: ["2026-11-01", "2026-11-02"], horizons: {
    5: {accuracy: .5, wait_errors: {absolute_minutes: {mean: 2.5}, mse_minutes_squared: 9, rmse_minutes: 3, count: 4, actual_numeric_wait_count: 6}},
    600: {accuracy: null, wait_errors: {absolute_minutes: null, rmse_minutes: null, count: 0, actual_numeric_wait_count: 0}},
  }});
  const html = ui.get("test-results").innerHTML;
  assert.match(html, /50\.0%/);
  assert.match(html, /2\.50/);
  assert.match(html, /MSE \(min²\)/);
  assert.match(html, /<td>9\.00<\/td>/);
  assert.match(html, /10 h/);
  assert.match(html, /N\/A/);
  assert.match(html, /2026-11-02/);
});

test("deleted models show recovery actions and block restore conflicts or busy jobs", () => {
  const ui = calendar([]);
  ui.renderDeletedModels([
    {id: "backup-a", name: "Saved model", deleted_at: "2026-11-21T12:00:00Z", scope: "run", can_restore: true},
    {id: "backup-b", name: "Existing file conflict", deleted_at: "2026-11-21T12:00:00Z", scope: "checkpoint", can_restore: false, reason: "Original exists"},
  ]);
  assert.equal(ui.get("deleted-model-count").textContent, 2);
  const rows = ui.get("deleted-model-list").children;
  assert.equal(rows[0].children[1].disabled, false);
  assert.equal(rows[1].children[1].disabled, true);
  assert.match(rows[1].children[0].children[1].textContent, /Original exists/);
  ui.state.jobBusy = true;
  ui.syncModelButtons();
  assert.equal(rows[0].children[1].disabled, true);
  ui.state.jobBusy = false;
  ui.syncModelButtons();
  assert.equal(rows[0].children[1].disabled, false);
  assert.equal(rows[1].children[1].disabled, true);
});

test("expanded training logs can return to the named model controls", () => {
  const ui = calendar([]);
  ui.showJobConsole(true);
  assert.equal(ui.get("studio-content").classList.contains("log-expanded"), true);
  assert.equal(ui.get("expand-log").textContent, "Show model controls");
  ui.showJobConsole(false);
  assert.equal(ui.get("studio-content").classList.contains("log-expanded"), false);
});

test("stop and discard is available only for unfinished training and disables while stopping", () => {
  const ui = calendar([]);
  for (const state of ["queued", "running"]) {
    ui.renderStopTraining({id: "job-a", action: "train", state});
    assert.equal(ui.get("stop-training").hidden, false);
    assert.equal(ui.get("stop-training").disabled, false);
    assert.equal(ui.state.jobId, "job-a");
  }
  ui.renderStopTraining({id: "job-a", action: "train", state: "cancelling"});
  assert.equal(ui.get("stop-training").disabled, true);
  assert.match(ui.get("stop-training").textContent, /Stopping/);
  for (const [action, state] of [["test", "running"], ["train", "complete"], ["train", "cancelled"]]) {
    ui.renderStopTraining({id: "job-a", action, state});
    assert.equal(ui.get("stop-training").hidden, true);
  }
});
