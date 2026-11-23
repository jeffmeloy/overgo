//go:build windows

package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/media"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserMOSSResumedRunReadsItsRequestBesideItsEnhancement holds recorded-request/do (SQA-470): a depth run
// submitted with a prompt enhancement as its source, the enhancement's identity sorting before its request's, is
// followed by a generation page that did not send it, and when it completes its comparison sets its picture against the
// control picture its recorded request names, the request told from the enhancement by its stored schema. The page had
// read the run's first JSON input, the enhancement, and offered no source.
func TestWebUIBrowserMOSSResumedRunReadsItsRequestBesideItsEnhancement(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the resumed request leg uses Chromium through cmd/webui-lane")
	hold, started, release := heldSession()
	fixture := newMOSSGenerationFixture(t, hold)
	testcheck.NoError(t, os.MkdirAll(filepath.Join(fixture.root, "adapters"), 0o755))
	for _, name := range []string{"8-steps-dmd2-lora", "moss_image_sft", "moss-control"} {
		testcheck.NoError(t, os.WriteFile(filepath.Join(fixture.root, "adapters", name+".safetensors"), nil, 0o600))
	}
	if registered := serveTestRequest(fixture.handler, http.MethodPost, "/library/register", `{"kind":"model","path":`+string(testcheck.Must(json.Marshal(fixture.root))(t))+`}`); registered.Code != http.StatusOK {
		t.Fatalf("register: %d %s", registered.Code, registered.Body)
	}
	canvas := image.NewRGBA(image.Rect(0, 0, 2, 2))
	canvas.Set(1, 0, color.RGBA{B: 255, A: 255})
	var encoded bytes.Buffer
	testcheck.NoError(t, png.Encode(&encoded, canvas))
	picture := artifact.Content{Descriptor: artifact.Descriptor{ID: testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, encoded.Bytes()))(t), Size: uint64(encoded.Len()), MediaType: media.PNGMediaType}, Data: encoded.Bytes()}
	testcheck.Must(artifact.Publish(t.Context(), fixture.store, artifact.Batch{Key: "fixture/source-a", Artifacts: []artifact.Descriptor{picture.Descriptor}, Contents: []artifact.Content{picture}}))(t)

	var listed []struct {
		Task   recipe.Task `json:"task"`
		Recipe artifact.ID `json:"recipe"`
	}
	capabilities := serveTestRequest(fixture.handler, http.MethodGet, "/generation/capabilities", "")
	testcheck.NoError(t, json.Unmarshal(capabilities.Body.Bytes(), &listed))
	if len(listed) == 0 {
		t.Fatalf("the generation workspace lists no capability: %s", capabilities.Body)
	}
	raw := testcheck.Must(json.Marshal(map[string]any{"preset": "ultra-depth-control", "prompt": "a lighthouse", "width": 1024, "height": 576,
		"control": map[string]any{"picture": picture.Descriptor.ID.String(), "strength": 1}}))(t)
	// The enhancement is made to sort before the run's request, as the stand-in session keeps it (newMOSSGenerationFixture).
	request := testcheck.Must(artifact.JSONContract(artifact.KindFile, "overgo.test-moss-request.v1").ContentBytes(raw))(t).Descriptor.ID
	var enhancement artifact.Content
	for nonce := 0; ; nonce++ {
		enhancement = testcheck.Must(artifact.JSONContent(artifact.JSONContract(artifact.KindFile, promptEnhancementSchema), promptEnhancement{
			Version: artifact.InitialDocumentVersion, Instruction: promptEnhanceInstruction, Original: "a lighthouse",
			Enhanced: fmt.Sprintf("a lighthouse on a rocky point at dusk, draft %d", nonce), Model: "stand-in"}))(t)
		if enhancement.Descriptor.ID.String() < request.String() {
			break
		}
	}
	testcheck.Must(artifact.Publish(t.Context(), fixture.store, artifact.Batch{Key: "fixture/enhancement", Contents: []artifact.Content{enhancement}}))(t)
	body := testcheck.Must(json.Marshal(map[string]any{"task": listed[0].Task, "recipe": listed[0].Recipe,
		"input": json.RawMessage(raw), "sources": []artifact.ID{enhancement.Descriptor.ID}}))(t)
	accepted := serveTestRequest(fixture.handler, http.MethodPost, "/generation/run", string(body))
	var submission workflowResponse
	if accepted.Code != http.StatusAccepted || json.Unmarshal(accepted.Body.Bytes(), &submission) != nil {
		t.Fatalf("the run was not accepted: %d %s", accepted.Code, accepted.Body)
	}
	select {
	case <-started:
	case <-t.Context().Done():
		t.Fatal("the run reached no session")
	}

	server := httptest.NewServer(fixture.handler)
	defer server.Close()
	defer fixture.handler.Close()
	_, browser, check, settle := openBrowserLeg(t, server.URL+"/#generation")
	defer browser.Close()
	settle(`!!document.querySelector('#panel-generation.active fieldset[data-control="control"]')`)
	check(`(() => {
  const panel=document.querySelector('#panel-generation');
  window.bench={
    panel,
    status:()=>panel.querySelector('[aria-label="run status"]'),
    button:()=>[...panel.querySelectorAll('button')].find(button=>button.textContent==='Run'),
    ran:()=>!!panel.querySelector('.workbench-output') && !bench.button().disabled,
    refused:()=>!bench.button().disabled && !panel.querySelector('.workbench-output') && bench.status().textContent.trim()!=='',
  };
  return true;
})()`)
	// The page follows the run it did not send.
	settle(`bench.button().disabled`)
	close(release)
	settle(`bench.ran() || bench.refused()`)
	check(`bench.ran() || (() => { throw new Error('the run was refused: ' + bench.status().textContent); })()`)
	var status operation.Status
	waited := serveTestRequest(fixture.handler, http.MethodGet, "/operations/wait?id="+submission.Operation.String(), "")
	testcheck.NoError(t, json.Unmarshal(waited.Body.Bytes(), &status))
	if status.Run == nil {
		t.Fatalf("the completed run names no record: %s", waited.Body)
	}
	var detail browseRunDetail
	testcheck.NoError(t, json.Unmarshal(serveTestRequest(fixture.handler, http.MethodGet, "/runs?id="+status.Run.String(), "").Body.Bytes(), &detail))
	if len(detail.Inputs) != 2 || detail.Inputs[0] != enhancement.Descriptor.ID {
		t.Fatalf("the run's inputs are %v; want the enhancement %s before its request", detail.Inputs, enhancement.Descriptor.ID)
	}
	a := string(testcheck.Must(json.Marshal(picture.Descriptor.ID.String()))(t))
	check(`(() => {
  const compare=bench.panel.querySelector('.ab-compare');
  if (!compare) throw new Error('the run followed opens no compare');
  const [under]=compare.querySelectorAll('select');
  if (under.value!==` + a + `) throw new Error('A is '+under.selectedOptions[0].textContent+'; want the picture its request started from');
  if (!under.selectedOptions[0].textContent.startsWith('started from')) throw new Error('A is labelled '+under.selectedOptions[0].textContent);
  return true;
})()`)
	check(`window.overgo.errors.length===0`)
	webuilane.Leg(t, "moss-resumed-request", "A MOSS depth run the generation page did not send, its prompt enhancement kept beside its request, is compared with the picture its recorded request names.")
}
