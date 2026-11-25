package webuilane

// idleScript runs in each document the browser loads, before the page's own scripts (Page.addScriptToEvaluateOnNewDocument),
// and keeps what the page is doing that can still change what it shows, so a wait whose predicate cannot come true ends
// once the page is idle, in every run, where it had waited for ever (idle-wait: four hangs on 2026-11-24). No clock
// says the page is idle: its own work does.
//
// The page is busy while it awaits a fetch to its body's end (a streamed answer included), a one-shot timer, a file
// read, a blob's bytes, a recording until it stops, an audio context's resumption, a picture or a clip loading, or a
// run it follows (overgo.awaiting). The activity stream, open for the page's life, is no work. The work an interval
// starts, synchronously in its handler, is polling: it does not make the page busy, but a wait does not end while it
// is in flight, and it ends only once every interval has fired since the page last was busy, so a poll's answer is
// read before the page is called idle. The wait keeps the native timers (__laneIdle.native) for its own loop.
const idleScript = `(() => {
  if (window.__laneIdle) return;
  const idle = window.__laneIdle = { work: 0, polling: 0, polls: 0, timeouts: new Set(), intervals: new Map() };
  const native = idle.native = {
    setTimeout: window.setTimeout.bind(window), clearTimeout: window.clearTimeout.bind(window),
    setInterval: window.setInterval.bind(window), clearInterval: window.clearInterval.bind(window), fetch: window.fetch.bind(window),
  };
  // begin counts one piece of work until the function it returns runs: polling work apart from the rest.
  idle.begin = () => {
    const polled = idle.polling > 0;
    if (polled) idle.polls++; else idle.work++;
    let ended = false;
    return () => { if (ended) return; ended = true; if (polled) idle.polls--; else idle.work--; };
  };
  window.setTimeout = function (handler, delay, ...rest) {
    if (idle.polling > 0) return native.setTimeout(handler, delay, ...rest);
    const id = native.setTimeout(() => { idle.timeouts.delete(id); if (typeof handler === "function") handler(...rest); }, delay);
    idle.timeouts.add(id);
    return id;
  };
  window.clearTimeout = function (id) { idle.timeouts.delete(id); native.clearTimeout(id); };
  window.setInterval = function (handler, delay, ...rest) {
    const entry = { fired: 0 };
    const id = native.setInterval(() => {
      entry.fired++;
      idle.polling++;
      try { if (typeof handler === "function") handler(...rest); } finally { idle.polling--; }
    }, delay);
    idle.intervals.set(id, entry);
    return id;
  };
  window.clearInterval = function (id) { idle.intervals.delete(id); native.clearInterval(id); };
  window.fetch = function (input, init) {
    const path = typeof input === "string" ? input : (input && input.url) || "";
    if (path.includes("/runtime/activity/stream")) return native.fetch(input, init);
    const end = idle.begin();
    return native.fetch(input, init).then((response) => {
      // A status that carries no body (204, 304) is passed on as it came, its work done.
      if (!response.body || [101, 103, 204, 205, 304].includes(response.status)) { end(); return response; }
      // A copy of the body is read to its end, so a streamed answer counts until it ends.
      const [kept, counted] = response.body.tee();
      const reader = counted.getReader();
      const drain = () => reader.read().then(({ done }) => (done ? end() : drain()), end);
      drain();
      const answer = new Response(kept, { status: response.status, statusText: response.statusText, headers: response.headers });
      Object.defineProperty(answer, "url", { value: response.url });
      return answer;
    }, (err) => { end(); throw err; });
  };
  // counted wraps a method whose work ends when its promise settles.
  const counted = (owner, name) => {
    const original = owner && owner[name];
    if (typeof original !== "function") return;
    owner[name] = function (...args) {
      const end = idle.begin();
      let answer;
      try { answer = original.apply(this, args); } catch (err) { end(); throw err; }
      Promise.resolve(answer).then(end, end);
      return answer;
    };
  };
  counted(Blob.prototype, "arrayBuffer");
  counted(Blob.prototype, "text");
  if (window.AudioContext) counted(AudioContext.prototype, "resume");
  if (window.AudioWorklet) counted(AudioWorklet.prototype, "addModule");
  // A file read counts until its loadend.
  for (const name of ["readAsArrayBuffer", "readAsDataURL", "readAsText", "readAsBinaryString"]) {
    const original = FileReader.prototype[name];
    if (typeof original !== "function") continue;
    FileReader.prototype[name] = function (...args) {
      const end = idle.begin();
      this.addEventListener("loadend", end, { once: true });
      return original.apply(this, args);
    };
  }
  // A recording counts from its start until its stop is dispatched; the page's own stop handler runs after this one.
  if (window.MediaRecorder) {
    const start = MediaRecorder.prototype.start;
    MediaRecorder.prototype.start = function (...args) {
      const end = idle.begin();
      this.addEventListener("stop", end, { once: true });
      this.addEventListener("error", end, { once: true });
      return start.apply(this, args);
    };
  }
  // A script or a stylesheet the page adds (boot.js loads its libraries so) loads until its load or error event, which
  // a listener on the document sees first, in its capture.
  const loaded = new WeakSet();
  for (const name of ["load", "error"]) {
    document.addEventListener(name, (event) => { if (event.target && event.target.tagName) loaded.add(event.target); }, true);
  }
  const loading = (element) => !loaded.has(element) && (element.tagName === "SCRIPT" ? !!element.src : element.rel === "stylesheet" && !element.sheet);
  // busy says the page awaits work of its own: a request, a timer, a file or blob read, a recording, a run it follows,
  // or a script, stylesheet, picture or clip still loading.
  idle.busy = () => document.readyState !== "complete" || idle.work > 0 || idle.timeouts.size > 0 ||
    ((window.overgo && window.overgo.awaiting) || 0) > 0 ||
    [...document.querySelectorAll("script[src], link[rel=stylesheet]")].some(loading) ||
    [...document.images].some((picture) => picture.src && !picture.complete) ||
    [...document.querySelectorAll("video, audio")].some((media) => media.currentSrc && media.readyState < 1 && !media.error);
  idle.fired = () => new Map([...idle.intervals].map(([id, entry]) => [id, entry.fired]));
})()`

// idleWait is the part of a wait (predicateWait) that ends it once the page is idle: it keeps the moment the page was
// last busy (the intervals' counts then), and the page is idle once it has not been busy since, no polling work is in
// flight and every interval has fired since that moment. A page without the instrumentation (idleScript) is never
// idle.
const idleWait = `
	const lane = window.__laneIdle;
	let quietFrom = null;
	const idle = () => {
		if (!lane) return false;
		if (lane.busy()) { quietFrom = null; return false; }
		const fired = lane.fired();
		if (!quietFrom) { quietFrom = fired; return false; }
		if (lane.polls > 0) return false;
		for (const [id, count] of fired) if (!quietFrom.has(id) || count <= quietFrom.get(id)) return false;
		return true;
	};
	const later = (lane && lane.native.setTimeout) || setTimeout;
`
