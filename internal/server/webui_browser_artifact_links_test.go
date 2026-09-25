package server

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"net/http/httptest"
	"os"
	"strconv"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserArtifactLinks opens stored artifacts with an API key set:
// a gallery link (evaluations and training cite their evidence this way)
// shows the entry inside the Artifacts tab, both when the tab first mounts
// and when it is already open, where it once opened the bearer route in a
// new tab that answered 401; the entry's image loads through the
// authenticated client, and Load lists the whole gallery again.
func TestWebUIBrowserArtifactLinks(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": artifact links run through cmd/webui-lane")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ids := make([]artifact.ID, 2)
	contents := make([]artifact.Content, 2)
	for index, shade := range []uint8{40, 200} {
		picture := image.NewGray(image.Rect(0, 0, 2, 2))
		picture.SetGray(0, 0, color.Gray{Y: shade})
		var encoded bytes.Buffer
		if err := png.Encode(&encoded, picture); err != nil {
			t.Fatal(err)
		}
		ids[index], err = artifact.IdentifyBytes(artifact.KindOutput, encoded.Bytes())
		if err != nil {
			t.Fatal(err)
		}
		contents[index] = artifact.Content{Descriptor: artifact.Descriptor{ID: ids[index], Size: uint64(encoded.Len()), MediaType: "image/png"}, Data: encoded.Bytes()}
	}
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: "server/artifact-links", Contents: contents}); err != nil {
		t.Fatal(err)
	}
	handler := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{}))
	handler.config.APIKey = testAPIKey
	server := httptest.NewServer(handler)
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	browser, err := webuilane.Open(ctx, path, server.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	check := func(expression string) { t.Helper(); assertBrowserPredicate(t, ctx, browser, expression) }
	settle := func(expression string) {
		t.Helper()
		if err := browser.Eventually(ctx, expression); err != nil {
			t.Fatalf("%s: %v", expression, err)
		}
	}
	settle(`!!document.querySelector('.composer textarea')`)
	check(`(() => { const key = document.getElementById('api-key'); key.value = ` + strconv.Quote(testAPIKey) + `; key.dispatchEvent(new Event('change')); return overgo.getKey() === key.value; })()`)
	// The page records every artifact read's status, so an unauthenticated one would show as 401.
	check(`(() => {
  window.laneArtifactStatuses = [];
  const original = window.fetch;
  window.fetch = function (path) {
    const answer = original.apply(this, arguments);
    if (typeof path === 'string' && path.startsWith('/artifacts')) answer.then((response) => window.laneArtifactStatuses.push(response.status), () => {});
    return answer;
  };
  return true;
})()`)
	shows := func(id artifact.ID) string {
		return `(() => {
  const panel = document.querySelector('#panel-artifacts.active');
  const entries = panel ? panel.querySelectorAll('.artifact-gallery article') : [];
  const image = entries.length === 1 && entries[0].querySelector('img');
  return entries.length === 1 && !!entries[0].querySelector('.mono[title=` + strconv.Quote(id.String()) + `]') &&
    !!image && image.src.startsWith('blob:') && image.complete && image.naturalWidth === 2;
})()`
	}
	open := func(id artifact.ID) {
		t.Helper()
		check(`(() => { const link = overgo.artifactLink(` + strconv.Quote(id.String()) + `, 'gallery entry', true); document.body.appendChild(link); link.click(); link.remove(); return true; })()`)
	}
	// The first link mounts the tab on the entry; the second reaches the open tab.
	open(ids[0])
	settle(shows(ids[0]))
	check(`location.hash === '#artifacts'`)
	open(ids[1])
	settle(shows(ids[1]))
	check(`(() => { const load = [...document.querySelectorAll('#panel-artifacts button')].find((button) => button.textContent === 'Load'); load.click(); return true; })()`)
	// The gallery also lists what the handler stored itself, so both entries are looked for by name.
	seeded := "[" + strconv.Quote(ids[0].String()) + ", " + strconv.Quote(ids[1].String()) + "]"
	settle(`(() => { const listed = [...document.querySelectorAll('#panel-artifacts .artifact-gallery article .mono')].map((node) => node.title); return ` + seeded + `.every((id) => listed.includes(id)); })()`)
	settle(`overgo.api.inFlight() === 0 && window.laneArtifactStatuses.length > 0 && window.laneArtifactStatuses.every((status) => status === 200)`)
	check(`!document.querySelector('#panel-artifacts .err-banner') && overgo.errors.length === 0`)
	webuilane.Leg(t, "artifact links leg", "gallery links opened two entries in the Artifacts tab (mounting and mounted) and every artifact read was authenticated")
}
