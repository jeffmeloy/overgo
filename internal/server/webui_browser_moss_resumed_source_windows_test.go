//go:build windows

package server

import (
	"bytes"
	"context"
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
	"overgo/internal/latentimage"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/testcheck"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserMOSSResumedRunComparesItsRecordedSource holds run-source-resumed/do (SQA-463): a depth section is run
// on MOSS's generation page with stored picture A as its control picture, the page is reloaded while the run draws and
// follows it again, and when the run completes its comparison sets its picture against A, labelled as the picture it
// started from, read from the request the run's record keeps. A run followed again had no input the page sent, so its
// comparison offered no source.
func TestWebUIBrowserMOSSResumedRunComparesItsRecordedSource(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the resumed source leg uses Chromium through cmd/webui-lane")
	started, release := make(chan struct{}), make(chan struct{})
	fixture := newMOSSGenerationFixture(t, func(ctx context.Context, _ artifact.Repository, _ modelrecipe.CapabilityEvidenceSelection, _ latentimage.Request) error {
		close(started)
		select {
		case <-release:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	})
	testcheck.NoError(t, os.MkdirAll(filepath.Join(fixture.root, "adapters"), 0o755))
	for _, name := range []string{"8-steps-dmd2-lora", "moss_image_sft", "moss-control"} {
		testcheck.NoError(t, os.WriteFile(filepath.Join(fixture.root, "adapters", name+".safetensors"), nil, 0o600))
	}
	registered := serveTestRequest(fixture.handler, http.MethodPost, "/library/register", `{"kind":"model","path":`+string(testcheck.Must(json.Marshal(fixture.root))(t))+`}`)
	if registered.Code != http.StatusOK {
		t.Fatalf("register: %d %s", registered.Code, registered.Body)
	}
	// A is a stored picture apart from the picture the stand-in session draws.
	canvas := image.NewRGBA(image.Rect(0, 0, 2, 2))
	canvas.Set(1, 0, color.RGBA{B: 255, A: 255})
	var encoded bytes.Buffer
	testcheck.NoError(t, png.Encode(&encoded, canvas))
	picture := artifact.Content{Descriptor: artifact.Descriptor{ID: testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, encoded.Bytes()))(t), Size: uint64(encoded.Len()), MediaType: media.PNGMediaType}, Data: encoded.Bytes()}
	testcheck.Must(artifact.Publish(t.Context(), fixture.store, artifact.Batch{Key: "fixture/source-a", Artifacts: []artifact.Descriptor{picture.Descriptor}, Contents: []artifact.Content{picture}}))(t)
	a := string(testcheck.Must(json.Marshal(picture.Descriptor.ID.String()))(t))

	server := httptest.NewServer(fixture.handler)
	defer server.Close()
	defer fixture.handler.Close()
	_, browser, check, settle := openBrowserLeg(t, server.URL+"/#generation")
	defer browser.Close()
	// bench reaches the generation page's controls; the page defines it again after its reload.
	const bench = `(() => {
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
  return true;
})()`
	settle(`!!document.querySelector('#panel-generation.active fieldset[data-control="control"]')`)
	check(bench)
	check(`(() => {
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
	// The page is reloaded while the run draws, and follows the run again.
	check(`(() => { setTimeout(() => location.reload(), 0); return true; })()`)
	settle(`!window.bench && !!document.querySelector('#panel-generation.active fieldset[data-control="control"]')`)
	check(bench)
	settle(`bench.button().disabled`)
	close(release)
	settle(`bench.ran() || bench.refused()`)
	check(`bench.ran() || (() => { throw new Error('the run was refused: ' + bench.status().textContent); })()`)
	check(`(() => {
  const compare=bench.panel.querySelector('.ab-compare');
  if (!compare) throw new Error('the run followed again opens no compare');
  const [under]=compare.querySelectorAll('select');
  if (under.value!==` + a + `) throw new Error('A is '+under.selectedOptions[0].textContent+'; want the picture the run started from');
  if (!under.selectedOptions[0].textContent.startsWith('started from')) throw new Error('A is labelled '+under.selectedOptions[0].textContent);
  return true;
})()`)
	check(`window.overgo.errors.length===0`)
	webuilane.Leg(t, "moss-resumed-source", "A MOSS depth run followed again after the generation page is reloaded is compared with the picture its recorded request names, labelled as its source.")
}
