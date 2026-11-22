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
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserMOSSSharpenedPair holds moss-sharpened-pair/page: each MOSS Ultra section saves its picture as drawn
// and as sharpened and its comparer shows the two (text to image 449 and 482 in 709), and a sharpened MOSS run on the
// generation page shows both, the drawn one first as its run publishes them, and opens its A/B compare with the drawn
// picture as A and the sharpened as B. A run that is not sharpened is shown and compared as before. A sharpened run had
// returned its sharpened picture alone.
func TestWebUIBrowserMOSSSharpenedPair(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the sharpened pair leg uses Chromium through cmd/webui-lane")
	fixture := newMOSSGenerationFixture(t, nil)
	canvas := image.NewRGBA(image.Rect(0, 0, 2, 2))
	canvas.Set(1, 1, color.RGBA{G: 255, A: 255})
	var drawn bytes.Buffer
	testcheck.NoError(t, png.Encode(&drawn, canvas))
	fixture.before = drawn.Bytes()
	drawnID := testcheck.Must(artifact.IdentifyBytes(artifact.KindOutput, drawn.Bytes()))(t)
	registered := serveTestRequest(fixture.handler, http.MethodPost, "/library/register", `{"kind":"model","path":`+string(testcheck.Must(json.Marshal(fixture.root))(t))+`}`)
	if registered.Code != http.StatusOK {
		t.Fatalf("register: %d %s", registered.Code, registered.Body)
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
    field:(name)=>[...panel.querySelectorAll('label.control')].find(label=>label.querySelector('span').textContent===name).querySelector('input,select,textarea'),
    button:()=>[...panel.querySelectorAll('button')].find(button=>button.textContent==='Run'),
  };
  const set=(field,value)=>{field.value=value;field.dispatchEvent(new Event('input',{bubbles:true}));};
  set(bench.field('prompt'),'a lighthouse');
  set(bench.field('width'),'256');
  set(bench.field('height'),'144');
  bench.button().click();
  return true;
})()`)
	settle(`!!bench.panel.querySelector('.workbench-output') && !bench.button().disabled`)
	check(`(() => {
  const shown=bench.panel.querySelectorAll('.workbench-output');
  if (shown.length!==2) throw new Error('the sharpened run shows '+shown.length+' pictures');
  const compare=bench.panel.querySelector('.ab-compare');
  if (!compare) throw new Error('the sharpened run opens no compare');
  const [a,b]=compare.querySelectorAll('select');
  if (a.value!==` + string(testcheck.Must(json.Marshal(drawnID.String()))(t)) + `) throw new Error('A is '+a.value+'; want the picture as drawn');
  if (b.value===a.value || !b.selectedOptions[0].textContent.startsWith('sharpened')) throw new Error('B is '+b.selectedOptions[0].textContent+'; want the sharpened picture');
  return a.selectedOptions[0].textContent.startsWith('drawn');
})()`)
	check(`window.overgo.errors.length===0`)
	webuilane.Leg(t, "moss-sharpened-pair", "A sharpened MOSS run on the generation page shows its picture as drawn and as sharpened and compares them, A drawn and B sharpened, as the MOSS Ultra workflow's comparers do.")
}
