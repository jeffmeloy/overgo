//go:build windows

package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/media"
	"overgo/internal/testcheck"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserMOSSLastPictureChosenIsSent holds moss-page/last-picture (SQA-489). A picture file dropped on an
// image to image section's picture is stored while a recent picture is chosen in the same slot. The recent one, the
// last choice, stays chosen and is the picture the run sends, and the dropped one is stored and listed among the
// recent. The upload's end had chosen the dropped picture over the later choice.
func TestWebUIBrowserMOSSLastPictureChosenIsSent(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the last-picture leg uses Chromium through cmd/webui-lane")
	fixture := newMOSSGenerationFixture(t, nil)
	registered := serveTestRequest(fixture.handler, http.MethodPost, "/library/register", `{"kind":"model","path":`+string(testcheck.Must(json.Marshal(fixture.root))(t))+`}`)
	if registered.Code != http.StatusOK {
		t.Fatalf("register: %d %s", registered.Code, registered.Body)
	}
	recent := tinyPNG(t).Data
	stored := artifact.Content{Descriptor: artifact.Descriptor{ID: testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, recent))(t), Size: uint64(len(recent)), MediaType: media.PNGMediaType}, Data: recent}
	testcheck.Must(artifact.Publish(t.Context(), fixture.store, artifact.Batch{Key: "fixture/last-picture", Artifacts: []artifact.Descriptor{stored.Descriptor}, Contents: []artifact.Content{stored}}))(t)
	drawn := image.NewRGBA(image.Rect(0, 0, 3, 3))
	drawn.Set(1, 1, color.RGBA{R: 30, G: 160, B: 90, A: 255})
	var encoded bytes.Buffer
	testcheck.NoError(t, png.Encode(&encoded, drawn))
	dropped := testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, encoded.Bytes()))(t)
	server := httptest.NewServer(fixture.handler)
	defer server.Close()
	defer fixture.handler.Close()
	_, browser, check, settle := openBrowserLeg(t, server.URL+"/#generation")
	defer browser.Close()
	settle(`document.querySelectorAll('#panel-generation .intake-thumb').length >= 3`)
	// The upload is held in the page until released, and meanwhile the recent picture is chosen.
	check(`(() => {
  const panel=document.querySelector('#panel-generation');
  window.uploadHeld=new Promise(resolve=>{ window.releaseUpload=resolve; });
  const original=window.fetch;
  window.fetch=function (path) { const answer=original.apply(this, arguments); return typeof path==='string' && path.startsWith('/artifacts/intake') ? window.uploadHeld.then(()=>answer) : answer; };
  const section=panel.querySelector('.workflow-task + .control-grid').firstElementChild.querySelector('select');
  section.value='ultra-image-to-image'; section.dispatchEvent(new Event('change',{bubbles:true}));
  window.source=[...panel.querySelectorAll('label.control')].find(label=>label.querySelector('span').textContent==='picture' && !label.closest('fieldset[data-control]'));
  const data=Uint8Array.from(atob(` + strconv.Quote(base64.StdEncoding.EncodeToString(encoded.Bytes())) + `), c=>c.charCodeAt(0));
  const held=new DataTransfer(); held.items.add(new File([data], 'dropped.png', {type:'image/png'}));
  source.querySelector('.picture-chooser').dispatchEvent(new DragEvent('drop', {dataTransfer:held, bubbles:true, cancelable:true}));
  [...source.querySelectorAll('.intake-thumb')].find(button=>button.dataset.id===` + strconv.Quote(stored.Descriptor.ID.String()) + `).click();
  window.releaseUpload();
  return true;
})()`)
	// The dropped picture is stored and listed among the recent in either case.
	settle(`[...source.querySelectorAll('.intake-thumb')].some(button=>button.dataset.id===` + strconv.Quote(dropped.String()) + `)`)
	check(`(() => {
  const panel=document.querySelector('#panel-generation');
  const prompt=[...panel.querySelectorAll('label.control')].find(label=>label.querySelector('span').textContent==='prompt').querySelector('textarea');
  prompt.value='a harbour at noon'; prompt.dispatchEvent(new Event('input',{bubbles:true}));
  window.runButton=[...panel.querySelectorAll('button')].find(button=>button.textContent==='Run');
  window.runButton.click();
  return true;
})()`)
	settle(`!!document.querySelector('#panel-generation .workbench-output') || (!window.runButton.disabled && document.querySelector('#panel-generation [aria-label="run status"]').textContent.trim()!=='')`)
	if inputs := fixture.requests(); len(inputs) != 1 || inputs[0].Image != stored.Descriptor.ID.String() {
		t.Fatalf("the run asked %+v; want the picture chosen last, %s, not the one stored meanwhile, %s", inputs, stored.Descriptor.ID, dropped)
	}
	check(`window.overgo.errors.length===0`)
	webuilane.Leg(t, "moss-last-picture", "A recent picture chosen while a dropped one is stored stays chosen and is sent, and the dropped one is listed among the recent.")

	// Two pictures dropped into the slot while both are stored: the first finishes first, and the last dropped is the
	// choice and is sent (moss-pictures/last-of-two, SQA-489). Both uploads had read the slot unchanged, so the first
	// to finish was chosen and the last dropped was not.
	pictureBytes := func(green uint8) ([]byte, artifact.ID) {
		drawn := image.NewRGBA(image.Rect(0, 0, 3, 3))
		drawn.Set(2, 0, color.RGBA{R: 200, G: green, B: 40, A: 255})
		var encoded bytes.Buffer
		testcheck.NoError(t, png.Encode(&encoded, drawn))
		return encoded.Bytes(), testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, encoded.Bytes()))(t)
	}
	first, firstID := pictureBytes(10)
	last, lastID := pictureBytes(220)
	check(`(() => {
  window.holds=[];
  const original=window.fetch;
  window.fetch=function (path) { const answer=original.apply(this, arguments); return typeof path==='string' && path.startsWith('/artifacts/intake') ? new Promise(resolve=>holds.push(resolve)).then(()=>answer) : answer; };
  const drop=(encoded, name)=>{
    const data=Uint8Array.from(atob(encoded), c=>c.charCodeAt(0));
    const held=new DataTransfer(); held.items.add(new File([data], name, {type:'image/png'}));
    source.querySelector('.picture-chooser').dispatchEvent(new DragEvent('drop', {dataTransfer:held, bubbles:true, cancelable:true}));
  };
  drop(` + strconv.Quote(base64.StdEncoding.EncodeToString(first)) + `, 'first.png');
  drop(` + strconv.Quote(base64.StdEncoding.EncodeToString(last)) + `, 'last.png');
  return true;
})()`)
	listed := func(id artifact.ID) string {
		return `[...source.querySelectorAll('.intake-thumb')].some(button=>button.dataset.id===` + strconv.Quote(id.String()) + `)`
	}
	settle(`window.holds.length===2`)
	check(`(() => { holds[0](); return true; })()`)
	settle(listed(firstID))
	check(`(() => { holds[1](); return true; })()`)
	settle(listed(lastID))
	check(`(() => {
  const panel=document.querySelector('#panel-generation');
  panel.querySelectorAll('.workbench-output').forEach(shown=>shown.remove());
  panel.querySelector('[aria-label="run status"]').replaceChildren();
  window.runButton.click();
  return true;
})()`)
	settle(`(!!document.querySelector('#panel-generation .workbench-output') && !window.runButton.disabled) || (!window.runButton.disabled && document.querySelector('#panel-generation [aria-label="run status"]').textContent.trim()!=='')`)
	if inputs := fixture.requests(); len(inputs) != 2 || inputs[1].Image != lastID.String() {
		t.Fatalf("the runs asked %+v; want the second to send the picture dropped last, %s, not the one that finished first, %s", inputs, lastID, firstID)
	}
	check(`window.overgo.errors.length===0`)
	webuilane.Leg(t, "moss-last-of-two", "Of two pictures dropped into one slot while both are stored, the one dropped last is chosen and sent, though the first finished first.")
}
