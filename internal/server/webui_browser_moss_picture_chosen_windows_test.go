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

// TestWebUIBrowserMOSSPictureIsChosenNotTyped holds moss-page/pictures (the owner's review of 2026-11-24, point 3):
// MOSS's picture slots (the picture a section starts from, its depth control's picture and depth map) take no typed
// text. Each shows its chosen picture's thumbnail and takes a picture file dropped on it or uploaded, stored and named
// by its identity, or a recent stored picture by its thumbnail. The control's picture and depth map had been text
// inputs, a stored identity typed in, beside a strip of recent pictures only for the top-level one.
func TestWebUIBrowserMOSSPictureIsChosenNotTyped(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the picture-chosen leg uses Chromium through cmd/webui-lane")
	fixture := newMOSSGenerationFixture(t, nil)
	registered := serveTestRequest(fixture.handler, http.MethodPost, "/library/register", `{"kind":"model","path":`+string(testcheck.Must(json.Marshal(fixture.root))(t))+`}`)
	if registered.Code != http.StatusOK {
		t.Fatalf("register: %d %s", registered.Code, registered.Body)
	}
	// A recent picture the store holds, and another the page is given as a file.
	recent := tinyPNG(t).Data
	stored := artifact.Content{Descriptor: artifact.Descriptor{ID: testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, recent))(t), Size: uint64(len(recent)), MediaType: media.PNGMediaType}, Data: recent}
	testcheck.Must(artifact.Publish(t.Context(), fixture.store, artifact.Batch{Key: "fixture/recent-picture", Artifacts: []artifact.Descriptor{stored.Descriptor}, Contents: []artifact.Content{stored}}))(t)
	drawn := image.NewRGBA(image.Rect(0, 0, 4, 3))
	for x := range 4 {
		drawn.Set(x, 1, color.RGBA{R: 200, G: 40, B: 90, A: 255})
	}
	var encoded bytes.Buffer
	testcheck.NoError(t, png.Encode(&encoded, drawn))
	dropped := testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, encoded.Bytes()))(t)
	server := httptest.NewServer(fixture.handler)
	defer server.Close()
	defer fixture.handler.Close()
	_, browser, check, settle := openBrowserLeg(t, server.URL+"/#generation")
	defer browser.Close()
	// The wait reads what a page without choosers shows too (each slot's recent picture), so the check fails there, not
	// waits.
	settle(`document.querySelectorAll('#panel-generation .intake-thumb').length >= 3`)
	check(`(() => {
  const panel=document.querySelector('#panel-generation');
  panel.querySelector('.advanced-controls').open=true;
  const named=(name, within)=>[...(within||panel).querySelectorAll('label.control')].find(label=>label.querySelector('span').textContent===name && (within || !label.closest('fieldset')));
  window.bench={
    panel,
    slot:{source:named('picture'), picture:named('picture', panel.querySelector('fieldset[data-control="control"]')), map:named('depth map', panel.querySelector('fieldset[data-control="control"]'))},
    file:(name)=>{const data=Uint8Array.from(atob(` + strconv.Quote(base64.StdEncoding.EncodeToString(encoded.Bytes())) + `), c=>c.charCodeAt(0)); const held=new DataTransfer(); held.items.add(new File([data], name, {type:'image/png'})); return held;},
    value:(slot)=>bench.slot[slot].querySelector('input[type=hidden]').value,
    shows:(slot, id)=>{const shown=bench.slot[slot].querySelector('.picture-thumb'); return !!shown && shown.title===id && shown.complete && shown.naturalWidth>0;},
  };
  for (const [slot, label] of Object.entries(bench.slot)) {
    if (!label || !label.querySelector('.picture-chooser')) throw new Error('the '+slot+' slot has no picture chooser');
    const typed=label.querySelectorAll('textarea, input:not([type=hidden]):not([type=file])');
    if (typed.length) throw new Error('the '+slot+' slot takes typed text: '+typed[0].outerHTML);
  }
  return true;
})()`)
	// A picture file dropped on the depth control's picture is stored and chosen by its identity.
	check(`(() => {
  const chooser=bench.slot.picture.querySelector('.picture-chooser');
  chooser.dispatchEvent(new DragEvent('drop', {dataTransfer:bench.file('dropped.png'), bubbles:true, cancelable:true}));
  return true;
})()`)
	settle(`bench.value('picture')===` + strconv.Quote(dropped.String()) + ` && bench.shows('picture', ` + strconv.Quote(dropped.String()) + `)`)
	// One uploaded as its depth map is too.
	check(`(() => {
  const upload=bench.slot.map.querySelector('input[type=file]');
  upload.files=bench.file('uploaded.png').files;
  upload.dispatchEvent(new Event('change', {bubbles:true}));
  return true;
})()`)
	settle(`bench.value('map')===` + strconv.Quote(dropped.String()) + ` && bench.shows('map', ` + strconv.Quote(dropped.String()) + `)`)
	// A recent picture is chosen by its thumbnail.
	check(`(() => {
  const thumb=[...bench.slot.source.querySelectorAll('.intake-thumb')].find(button=>button.dataset.id===` + strconv.Quote(stored.Descriptor.ID.String()) + `);
  if (!thumb) throw new Error('the recent picture is not offered');
  thumb.click();
  return true;
})()`)
	settle(`bench.value('source')===` + strconv.Quote(stored.Descriptor.ID.String()) + ` && bench.shows('source', ` + strconv.Quote(stored.Descriptor.ID.String()) + `)`)
	check(`window.overgo.errors.length===0`)
	webuilane.Leg(t, "moss-picture-chosen", "MOSS's picture, depth-control picture and depth map are chosen by a dropped file, an upload or a recent picture's thumbnail, each shown by its thumbnail, and none takes typed text.")
}
