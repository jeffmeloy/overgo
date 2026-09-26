package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/png"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserFrontSurfaces drives the front page's shell over a stored
// chat model: the welcome card's starters focus the composer and open the
// file picker, the status dots name the server and the device, the composer
// highlights a drag and takes a dropped Markdown and PDF file whose
// extracted text reaches the model, and the turn's inspector reports its
// status and embeds an analysis view seeded with the turn.
func TestWebUIBrowserFrontSurfaces(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the front surfaces leg uses Chromium through cmd/webui-lane")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	handler := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{pieces: []string{"Front answer"}}))
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
	settle(`!!document.querySelector('.composer textarea') && !!document.querySelector('.front-empty .starters')`)
	check(`(() => {
  window.frontButton = (label, root) => [...(root || document).querySelectorAll('button')].find(button => button.textContent === label && !button.hidden);
  window.frontPickerOpened = false;
  document.querySelector('.composer input[type=file]').addEventListener('click', event => { frontPickerOpened = true; event.preventDefault(); });
  return true;
})()`)

	// The welcome card's starters: the question focuses the composer, the file opens the picker.
	check(`(() => { frontButton('Ask a question', document.querySelector('.front-empty')).click(); return document.activeElement === document.querySelector('.composer textarea'); })()`)
	check(`(() => { frontButton('Attach a file', document.querySelector('.front-empty')).click(); return frontPickerOpened; })()`)

	// The status dots name the server and the device.
	settle(`document.getElementById('server-dot').title === 'server online' && document.getElementById('device-dot').title.startsWith('device peak ')`)

	// The composer highlights a drag and takes dropped files whose text reaches the model.
	pdf := base64.StdEncoding.EncodeToString(testPDF(t, "BT /F1 12 Tf 72 700 Td (Front PDF line) Tj ET"))
	check(`(() => {
  const composer = document.querySelector('.composer');
  const transfer = new DataTransfer();
  transfer.items.add(new File(['# Front notes\nbody'], 'notes.md', { type: 'text/markdown' }));
  transfer.items.add(new File([Uint8Array.from(atob(` + strconv.Quote(pdf) + `), char => char.charCodeAt(0))], 'paper.pdf', { type: 'application/pdf' }));
  composer.dispatchEvent(new DragEvent('dragover', { dataTransfer: transfer, bubbles: true, cancelable: true }));
  window.frontHighlighted = composer.classList.contains('drop');
  composer.dispatchEvent(new DragEvent('drop', { dataTransfer: transfer, bubbles: true, cancelable: true }));
  return frontHighlighted && !composer.classList.contains('drop');
})()`)
	settle(`[...document.querySelectorAll('.attachment-row')].length === 2 && [...document.querySelectorAll('.attachment-row')].every(row => row.dataset.state === 'ready')`)
	check(`(() => {
  const input = document.querySelector('.composer textarea');
  input.value = 'summarize'; input.dispatchEvent(new Event('input', { bubbles: true }));
  document.querySelector('.send-button').click();
  return true;
})()`)
	settle(`[...document.querySelectorAll('.msg')].some(message => message.textContent.includes('Front answer'))`)
	var listed conversationListResponse
	if err := json.Unmarshal(serveTestRequest(handler, http.MethodGet, "/interactions", "").Body.Bytes(), &listed); err != nil || len(listed.Conversations) != 1 {
		t.Fatalf("conversations = %+v, %v", listed, err)
	}
	var chain conversationMessagesResponse
	if err := json.Unmarshal(serveTestRequest(handler, http.MethodGet, "/interactions/messages?response="+listed.Conversations[0].Latest, "").Body.Bytes(), &chain); err != nil || len(chain.Messages) == 0 {
		t.Fatalf("chain = %+v, %v", chain, err)
	}
	for _, text := range []string{`[file name="notes.md"]`, "# Front notes\nbody", `[file name="paper.pdf"]`, "Front PDF line"} {
		if !strings.Contains(chain.Messages[0].Content, text) {
			t.Fatalf("the dropped files did not reach the model: %q lacks %q", chain.Messages[0].Content, text)
		}
	}

	// An image past the served dimension limit is refused where it was attached.
	var wide bytes.Buffer
	if err := png.Encode(&wide, image.NewRGBA(image.Rect(0, 0, 2, 2))); err != nil {
		t.Fatal(err)
	}
	check(`(() => {
  window.frontDimension = overgo.capabilities().media.max_image_dimension;
  overgo.capabilities().media.max_image_dimension = 1;
  const transfer = new DataTransfer();
  transfer.items.add(new File([Uint8Array.from(atob(` + strconv.Quote(base64.StdEncoding.EncodeToString(wide.Bytes())) + `), char => char.charCodeAt(0))], 'wide.png', { type: 'image/png' }));
  const picker = document.querySelector('.composer input[type=file]');
  picker.files = transfer.files; picker.dispatchEvent(new Event('change'));
  return true;
})()`)
	settle(`[...document.querySelectorAll('.attachment-row')].some(row => row.querySelector('.attachment-name').textContent === 'wide.png' && row.dataset.state === 'refused')`)
	check(`(() => { overgo.capabilities().media.max_image_dimension = frontDimension; document.querySelector('[aria-label="Remove wide.png"]').click(); return true; })()`)

	// The turn's inspector reports its status and embeds a view seeded with the turn.
	check(`(() => { [...document.querySelectorAll('.msg button.link-button')].find(button => button.textContent === 'inspect').click(); return true; })()`)
	settle(`!document.getElementById('inspector').hidden && document.getElementById('inspector').textContent.includes('Status') && !!frontButton('Logits', document.getElementById('inspector'))`)
	check(`(() => { frontButton('Logits', document.getElementById('inspector')).click(); return true; })()`)
	settle(`(document.querySelector('#inspector [aria-label="Prompt to analyze"]')?.value || '').includes('Front answer')`)
	webuilane.Leg(t, "front surfaces leg", "the welcome starters focused the composer and opened the picker, the dots named the server and device, dropped Markdown and PDF files reached the model, an image past the dimension limit was refused, and the turn's inspector embedded a view seeded with the turn")
}
