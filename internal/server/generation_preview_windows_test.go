//go:build windows

package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/webuilane"
)

// TestTheGenerationPagePreviewsWhatARunReads holds the generation workspace's preview (moss-ultra-compare/depth-preview):
// a page's request to a listed capability that previews is answered with the stored picture its preview made (a MOSS
// depth section's map), the preview asked the request as the page sent it at the model's path; a request with a field
// no control declares, or a page that is none, is refused before any preview.
func TestTheGenerationPagePreviewsWhatARunReads(t *testing.T) {
	t.Parallel()
	fixture := newMOSSGenerationFixture(t, nil)
	var receipt struct{ Recipe artifact.ID }
	registered := serveTestRequest(fixture.handler, http.MethodPost, "/library/register", `{"kind":"model","path":`+string(testcheck.Must(json.Marshal(fixture.root))(t))+`}`)
	if registered.Code != http.StatusOK {
		t.Fatalf("register: %d %s", registered.Code, registered.Body)
	}
	testcheck.NoError(t, json.Unmarshal(registered.Body.Bytes(), &receipt))
	picture := testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, []byte("a picture")))(t)
	input := `{"prompt":"a fox","width":32,"height":48,"control":{"picture":"` + picture.String() + `","end":0.5}}`
	preview := func(input string, limit int) *httptest.ResponseRecorder {
		body := testcheck.Must(json.Marshal(map[string]any{"task": recipe.TaskImageGen, "recipe": receipt.Recipe, "input": json.RawMessage(input), "position": 0, "limit": limit}))(t)
		return serveTestRequest(fixture.handler, http.MethodPost, "/generation/preview", string(body))
	}
	answered := preview(input, 1)
	if answered.Code != http.StatusOK {
		t.Fatalf("preview: %d %s", answered.Code, answered.Body)
	}
	var shown generationPreview
	testcheck.NoError(t, json.Unmarshal(answered.Body.Bytes(), &shown))
	content, present, err := artifact.ReadContent(t.Context(), fixture.store, shown.Output)
	if err != nil || !present || shown.MediaType != "image/png" || !bytes.Equal(content.Data, tinyPNG(t).Data) {
		t.Fatalf("the preview answered %+v, stored %t (%v); want the preview's PNG stored", shown, present, err)
	}
	if len(fixture.previewed) != 1 || fixture.previewed[0] != input {
		t.Errorf("the preview was asked %q; want the page's request as it sent it", fixture.previewed)
	}
	for name, refused := range map[string]*httptest.ResponseRecorder{
		"an undeclared field": preview(`{"prompt":"a fox","mu":1}`, 1),
		"no page":             preview(input, 0),
	} {
		if refused.Code != http.StatusBadRequest {
			t.Errorf("a preview with %s answered %d %s; want it refused", name, refused.Code, refused.Body)
		}
	}
	if len(fixture.previewed) != 1 {
		t.Errorf("a refused preview reached the capability: %q", fixture.previewed)
	}
}

// TestWebUIBrowserDepthPreview drives the generation page's depth preview (moss-ultra-compare/depth-preview): MOSS's page,
// which holds a depth control, offers the preview; given a stored picture in the control, the preview shows the picture
// the workspace stored of the request the page composed.
func TestWebUIBrowserDepthPreview(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the depth preview leg uses Chromium through cmd/webui-lane")
	fixture := newMOSSGenerationFixture(t, nil)
	registered := serveTestRequest(fixture.handler, http.MethodPost, "/library/register", `{"kind":"model","path":`+string(testcheck.Must(json.Marshal(fixture.root))(t))+`}`)
	if registered.Code != http.StatusOK {
		t.Fatalf("register: %d %s", registered.Code, registered.Body)
	}
	picture := testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, []byte("a picture")))(t)
	server := httptest.NewServer(fixture.handler)
	defer server.Close()
	defer fixture.handler.Close()
	_, browser, check, settle := openBrowserLeg(t, server.URL+"/#generation")
	defer browser.Close()
	settle(`!!document.querySelector('#panel-generation.active fieldset[data-control="control"]')`)
	check(`(() => {
  const panel=document.querySelector('#panel-generation');
  const offered=[...panel.querySelectorAll('button')].find(button=>button.textContent==='Preview depth map');
  if (!offered || offered.closest('[hidden]')) return false;
  const prompt=[...panel.querySelectorAll('label.control')].find(label=>label.querySelector('span').textContent==='prompt').querySelector('textarea');
  prompt.value='a fox';
  const slot=[...panel.querySelectorAll('fieldset[data-control="control"] label.control')].find(label=>label.querySelector('span').textContent==='picture').querySelector('input');
  slot.value='` + picture.String() + `';
  offered.click();
  return true;
})()`)
	settle(`!!document.querySelector('#panel-generation .depth-preview img')`)
	settle(`(() => { const shown=document.querySelector('#panel-generation .depth-preview img'); return shown.src.startsWith('blob:') && shown.complete && shown.naturalWidth===2; })()`)
	fixture.mu.Lock()
	previewed := append([]string(nil), fixture.previewed...)
	fixture.mu.Unlock()
	if len(previewed) != 1 || !bytes.Contains([]byte(previewed[0]), []byte(`"control":{"picture":"`+picture.String()+`"}`)) {
		t.Fatalf("the page asked the preview %q; want its request with the control's picture", previewed)
	}
	check(`window.overgo.errors.length===0`)
	webuilane.Leg(t, "depth-preview", "MOSS's generation page offers a depth preview beside its depth control and shows the picture the workspace stored of the page's request.")
}
