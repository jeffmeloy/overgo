//go:build windows

package server

import (
	"bytes"
	"encoding/json"
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
	"overgo/internal/testcheck"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserMOSSComparesTheSourceItsRunStartedFrom holds moss-workflow-surface/source (SQA-463): a depth section
// is run on MOSS's generation page with stored picture A as its control picture, the form's control picture is set to
// stored picture B while the run draws, and when the run completes its comparison sets its picture against A, labelled
// as the picture it started from, and offers B as no source. The page had read the source from the form as the run
// ended, so it labelled B the run's source and compared the run with it, and did not offer A.
func TestWebUIBrowserMOSSComparesTheSourceItsRunStartedFrom(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the run source leg uses Chromium through cmd/webui-lane")
	hold, started, release := heldSession()
	fixture := newMOSSGenerationFixture(t, hold)
	testcheck.NoError(t, os.MkdirAll(filepath.Join(fixture.root, "adapters"), 0o755))
	for _, name := range []string{"8-steps-dmd2-lora", "moss_image_sft", "moss-control"} {
		testcheck.NoError(t, os.WriteFile(filepath.Join(fixture.root, "adapters", name+".safetensors"), nil, 0o600))
	}
	registered := serveTestRequest(fixture.handler, http.MethodPost, "/library/register", `{"kind":"model","path":`+string(testcheck.Must(json.Marshal(fixture.root))(t))+`}`)
	if registered.Code != http.StatusOK {
		t.Fatalf("register: %d %s", registered.Code, registered.Body)
	}
	// A and B are stored pictures apart from each other and from the picture the stand-in session draws.
	stored := func(key string, shade color.RGBA) string {
		canvas := image.NewRGBA(image.Rect(0, 0, 2, 2))
		canvas.Set(1, 0, shade)
		var encoded bytes.Buffer
		testcheck.NoError(t, png.Encode(&encoded, canvas))
		picture := artifact.Content{Descriptor: artifact.Descriptor{ID: testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, encoded.Bytes()))(t), Size: uint64(encoded.Len()), MediaType: media.PNGMediaType}, Data: encoded.Bytes()}
		testcheck.Must(artifact.Publish(t.Context(), fixture.store, artifact.Batch{Key: key, Artifacts: []artifact.Descriptor{picture.Descriptor}, Contents: []artifact.Content{picture}}))(t)
		return picture.Descriptor.ID.String()
	}
	idA, idB := stored("fixture/source-a", color.RGBA{B: 255, A: 255}), stored("fixture/source-b", color.RGBA{G: 255, A: 255})
	a, b := string(testcheck.Must(json.Marshal(idA))(t)), string(testcheck.Must(json.Marshal(idB))(t))

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
    field:(name)=>[...panel.querySelectorAll('label.control')].find(label=>label.querySelector('span').textContent===name).querySelector('input,select,textarea'),
    part:(group,name)=>[...panel.querySelector('fieldset[data-control="'+group+'"]').querySelectorAll('label.control')].find(label=>label.querySelector('span').textContent===name).querySelector('input,select,textarea'),
    set:(field,value)=>{field.value=value;field.dispatchEvent(new Event('input',{bubbles:true}));},
    status:()=>panel.querySelector('[aria-label="run status"]'),
    button:()=>[...panel.querySelectorAll('button')].find(button=>button.textContent==='Run'),
    ran:()=>!!panel.querySelector('.workbench-output') && !bench.button().disabled,
    refused:()=>!bench.button().disabled && !panel.querySelector('.workbench-output') && bench.status().textContent.trim()!=='',
  };
  panel.querySelector('.advanced-controls').open=true;
  const preset=bench.field('preset');
  preset.value='ultra-depth-control';
  preset.dispatchEvent(new Event('change',{bubbles:true}));
  bench.set(bench.field('prompt'),'a lighthouse');
  bench.set(bench.field('width'),'1024');
  bench.set(bench.field('height'),'576');
  bench.set(bench.part('control','picture'),` + a + `);
  bench.set(bench.part('control','strength'),'1');
  bench.button().click();
  return true;
})()`)
	select {
	case <-started:
	case <-t.Context().Done():
		t.Fatal("the run reached no session")
	}
	// The form changes while the run draws.
	check(`(() => { bench.set(bench.part('control','picture'),` + b + `); return true; })()`)
	close(release)
	settle(`bench.ran() || bench.refused()`)
	check(`bench.ran() || (() => { throw new Error('the run was refused: ' + bench.status().textContent); })()`)
	if inputs := fixture.requests(); len(inputs) != 1 || inputs[0].Control == nil || inputs[0].Control.Picture != idA {
		t.Fatalf("the session was sent %+v; want one request with A as its control picture", inputs)
	}
	check(`(() => {
  const compare=bench.panel.querySelector('.ab-compare');
  if (!compare) throw new Error('the run opens no compare');
  const [under]=compare.querySelectorAll('select');
  if (under.value!==` + a + `) throw new Error('A is '+under.selectedOptions[0].textContent+'; want the picture the run started from');
  if (!under.selectedOptions[0].textContent.startsWith('started from')) throw new Error('A is labelled '+under.selectedOptions[0].textContent);
  const offered=[...under.options].find(option=>option.value===` + b + `);
  if (offered) throw new Error('the compare offers the form\'s later picture as '+offered.textContent);
  return true;
})()`)
	check(`window.overgo.errors.length===0`)
	webuilane.Leg(t, "moss-run-source", "A MOSS depth run on the generation page is compared with the picture it was sent with, labelled as its source, though the form's picture changed while it drew.")
}
