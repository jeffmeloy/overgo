package server

import (
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserPeerWorkspace drives the peers workspace: the enrolled
// peer lists in the inventory and opens its detail, draining it moves its
// state, the enrollment and placement forms render from their declared
// schemas (an identity field of its kind, a task choice, integer bounds and
// units in the labels) and refuse to submit what the schema rejects, and the
// placement form's bounds hold a replica count below its minimum.
func TestWebUIBrowserPeerWorkspace(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the peer workspace leg uses Chromium through cmd/webui-lane")
	}
	fixture := newPeerWorkspaceFixture(t, "peer-browser", "")
	publishPeerControlFixture(t, fixture, "browser-peer")
	server := httptest.NewServer(fixture.handler)
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
	check(`(() => { location.hash = 'peers'; return true; })()`)
	settle(`!!document.querySelector('#panel-peers.active')`)
	check(`(() => {
  window.peersPanel = () => document.querySelector('#panel-peers');
  window.peersButton = (label) => [...peersPanel().querySelectorAll('button')].find(button => button.textContent === label);
  window.peersField = (label) => peersPanel().querySelector('[aria-label="' + label + '"]');
  window.peersStatus = () => peersPanel().querySelector('.note').textContent;
  return true;
})()`)

	// The enrolled peer lists and opens its detail; draining moves its state.
	settle(`!!peersButton('browser-peer')`)
	check(`(() => { peersButton('browser-peer').closest('tr').click(); return true; })()`)
	settle(`peersPanel().textContent.includes('environment / ') && !!peersButton('Drain')`)
	check(`(() => { peersButton('Drain').click(); return true; })()`)
	settle(`peersStatus().startsWith('draining / ')`)
	settle(`[...peersPanel().querySelectorAll('table tr')].some(row => row.textContent.includes('browser-peer') && row.textContent.includes('draining'))`)

	// The forms render from their schemas.
	check(`peersField('Model').getAttribute('data-identity-kind') === 'model' && peersField('Model').required`)
	check(`peersField('Task').tagName === 'SELECT' && [...peersField('Task').options].some(option => option.value === 'generation')`)
	check(`peersField('Minimum replicas').type === 'number' && peersField('Minimum replicas').min === '1'`)
	check(`[...peersPanel().querySelectorAll('label span')].some(span => span.textContent === 'Maximum measured latency (ns)') &&
  [...peersPanel().querySelectorAll('label span')].some(span => span.textContent === 'Maximum device memory (bytes)')`)

	// A form the schema rejects does not submit.
	check(`(() => { peersButton('Enroll approved peer').click(); return true; })()`)
	settle(`peersStatus() === 'Complete every enrollment field'`)
	check(`(() => { peersButton('Compile placement').click(); return true; })()`)
	settle(`peersStatus() === 'Complete every applicable placement field'`)
	check(`(() => {
  const field = peersField('Model');
  field.value = 'model:sha256:' + '0'.repeat(64); field.dispatchEvent(new Event('input', { bubbles: true }));
  const replicas = peersField('Minimum replicas');
  replicas.value = '0'; replicas.dispatchEvent(new Event('input', { bubbles: true }));
  return !replicas.checkValidity();
})()`)
	check(`(() => { peersStatus(); peersButton('Compile placement').click(); return true; })()`)
	settle(`peersStatus() === 'Complete every applicable placement field'`)
	webuilane.Leg(t, "peer workspace leg", "an enrolled peer listed, opened and drained, and the enrollment and placement forms rendered from their schemas and refused what the schemas reject")
}
