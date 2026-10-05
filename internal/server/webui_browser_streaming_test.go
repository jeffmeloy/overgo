package server

import (
	"net/http/httptest"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserStreamingComposerAppendsInPlace holds the streaming
// composer to what a long answer costs and keeps: a profile of the page
// (2026-10-05) found 94 percent of an 8000-token answer's 5.6 s of main
// thread in the layout scroll() forced after every token, so a burst of tokens
// must append its text and follow the bottom once per frame, build the role
// header once while streaming and once at the end, and never replace the text
// it has already shown. It keeps the exact text, a reader's scroll position
// when they have scrolled up, a stopped turn's note and announcement, and the
// log's accessibility, and a disposed thread writes nothing after it.
func TestWebUIBrowserStreamingComposerAppendsInPlace(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "streaming composer runs through cmd/webui-lane")
	repository, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = repository.Close() })
	handler := newTestHandlerForRepository(t, repository, responseRecipeGenerator(t, &fakeGenerator{}))
	defer handler.Close()
	server := httptest.NewServer(handler)
	defer server.Close()
	_, browser, check, settle := openBrowserLeg(t, server.URL+"/")
	defer browser.Close()
	settle(`!!window.overgo && typeof window.overgo.thread === "function" && window.overgo.errors.length === 0`)
	check(`(async () => {
  const frames = () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  const words = (count) => Array.from({ length: count }, (_, index) => "word" + index + (index % 9 === 8 ? "\n" : " "));
  const events = (pieces, last) => (async function* () { for (const text of pieces) yield { type: "token", text }; yield last || { type: "done" }; })();
  // open builds a thread in a fixed box and counts what streaming does to it.
  const open = () => {
    const host = document.createElement("div");
    host.style.cssText = "position:fixed;left:0;top:0;width:480px;height:300px;overflow:hidden;z-index:99999;background:#fff";
    document.body.appendChild(host);
    const thread = overgo.thread(host, {});
    const log = host.querySelector(".chat-log");
    log.style.cssText = "height:300px;overflow-y:auto";
    const seen = { writes: 0, headers: 0 };
    const own = Object.getOwnPropertyDescriptor(Element.prototype, "scrollTop");
    Object.defineProperty(log, "scrollTop", { configurable: true, get() { return own.get.call(this); }, set(value) { seen.writes++; own.set.call(this, value); } });
    new MutationObserver((records) => { for (const record of records) for (const node of record.addedNodes) if (node.classList && node.classList.contains("role")) seen.headers++; }).observe(host, { childList: true, subtree: true });
    return { host, thread, log, seen, announcement: host.querySelector(".sr-only") };
  };
  const atBottom = (log) => Math.ceil(log.scrollTop + log.clientHeight) >= log.scrollHeight;

  // A burst of tokens in one task: the text is exact, the role header is built a few times, not once per token,
  // the scroll position is written a few times, and the log still follows the bottom.
  const burst = open();
  const pieces = words(3000);
  await burst.thread.consume(events(pieces), null);
  await frames();
  const message = burst.thread.messages[burst.thread.messages.length - 1];
  if (message.content !== pieces.join("") || burst.log.querySelector(".msg.assistant").textContent.indexOf("word2999") < 0) throw new Error("the streamed text is not exact");
  if (burst.seen.headers > 4) throw new Error("the role header was rebuilt " + burst.seen.headers + " times for 3000 tokens");
  if (burst.seen.writes > 8) throw new Error("the scroll position was written " + burst.seen.writes + " times for 3000 tokens");
  if (!atBottom(burst.log)) throw new Error("the log did not follow the bottom");
  if (burst.log.getAttribute("role") !== "log" || burst.log.getAttribute("aria-live") !== "off" || burst.announcement.textContent !== "Response ready.") throw new Error("the log lost its accessibility");

  // A reader who scrolled up keeps their place while the answer grows, and the log offers the way back.
  const reader = open();
  await reader.thread.consume(events(words(400)), null);
  await frames();
  reader.log.scrollTop = 0;
  await frames();
  const held = reader.thread.messages[reader.thread.messages.length - 1];
  await reader.thread.consume(events(words(600)), held);
  await frames();
  if (reader.log.scrollTop !== 0) throw new Error("a reader's scroll position moved to " + reader.log.scrollTop);
  if (reader.host.querySelector(".jump-latest").hidden) throw new Error("the way back to the latest was not offered");

  // A stopped turn keeps the text it showed, says so, and a disposed thread writes nothing after it.
  const stopped = open();
  const some = words(50);
  await stopped.thread.consume(events(some, { type: "cancelled" }), null);
  const stoppedMessage = stopped.thread.messages[stopped.thread.messages.length - 1];
  if (stoppedMessage.content !== some.join("") || stopped.log.textContent.indexOf("Stopped") < 0 || stopped.announcement.textContent !== "Response stopped.") throw new Error("a stopped turn lost its text, note or announcement");
  stopped.thread.dispose();
  const written = stopped.seen.writes;
  await frames();
  if (stopped.seen.writes !== written) throw new Error("a disposed thread wrote its scroll position");
  for (const view of [burst, reader, stopped]) view.host.remove();
  return true;
})()`)
	webuilane.Leg(t, "streaming composer leg", "a burst of 3000 tokens appended exact text, built the role header a few times and wrote the scroll position a few times and still followed the bottom; a reader who scrolled up kept their place and the way back; a stopped turn kept its text, note and announcement and a disposed thread wrote nothing")
}
