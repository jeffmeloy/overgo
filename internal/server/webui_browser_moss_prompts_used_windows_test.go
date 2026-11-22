//go:build windows

package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"overgo/internal/latentimage"
	"overgo/internal/testcheck"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserMOSSPromptsUsed holds moss-prompts-used/page: a MOSS run on the generation page shows the prompt and
// the negative prompt its picture was drawn from, as the MOSS Ultra workflow shows them beside each section's picture
// ("Prompt used", "Negative prompt used"), from the document its run publishes beside the picture, and offers each back
// as the manual prompt: the prompt mode text, which sends no layout, the text in the prompt or negative prompt, so the next
// run draws it as written. A run's picture is still shown and compared as before. The page had shown a run's pictures
// alone, and the prompts an enhancer wrote or a layout made were nowhere on it.
func TestWebUIBrowserMOSSPromptsUsed(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the prompts used leg uses Chromium through cmd/webui-lane")
	fixture := newMOSSGenerationFixture(t, nil)
	const written, excluded = "a red fox resting in fresh snow beside a birch forest at first light", "blurry edges, washed out colours"
	// The stand-in reports what an enhancer wrote for an enhanced request, and the request's own text otherwise.
	fixture.drawn = func(request latentimage.Request) (string, string) {
		if request.Prompt == "a fox" {
			return written, excluded
		}
		return request.Prompt, request.NegativePrompt
	}
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
    button:(text)=>[...panel.querySelectorAll('button')].find(button=>button.textContent===text),
    used:()=>panel.querySelector('section.prompts-used'),
  };
  panel.querySelector('.advanced-controls').open=true;
  const set=(field,value)=>{field.value=value;field.dispatchEvent(new Event('input',{bubbles:true}));};
  set(bench.field('prompt'),'a fox');
  set(bench.field('width'),'256');
  set(bench.field('height'),'144');
  bench.button('Run').click();
  return true;
})()`)
	settle(`!!bench.panel.querySelector('.workbench-output') && !bench.button('Run').disabled`)
	check(`(() => {
  if (!bench.used()) throw new Error('the run shows no prompts used');
  const shown=[...bench.used().querySelectorAll('.prompt-used')].map(part=>[part.querySelector('.section-title').textContent, part.querySelector('pre').textContent]);
  if (JSON.stringify(shown)!==JSON.stringify([['Prompt used',` + string(testcheck.Must(json.Marshal(written))(t)) + `],['Negative prompt used',` + string(testcheck.Must(json.Marshal(excluded))(t)) + `]]))
    throw new Error('the run shows '+JSON.stringify(shown));
  if (bench.panel.querySelectorAll('.workbench-output').length!==1) throw new Error('the run shows its document as a picture');
  bench.button('Use as the prompt').click();
  bench.button('Use as the negative prompt').click();
  return bench.field('prompt').value===` + string(testcheck.Must(json.Marshal(written))(t)) + ` && bench.field('negative prompt').value===` + string(testcheck.Must(json.Marshal(excluded))(t)) + ` && bench.field('prompt mode').value==='text';
})()`)
	// The prompt offered back is drawn as written: the next run's request is it.
	check(`(() => { bench.panel.querySelectorAll('.workbench-output').forEach(shown=>shown.remove()); bench.button('Run').click(); return true; })()`)
	settle(`!!bench.panel.querySelector('.workbench-output') && !bench.button('Run').disabled`)
	inputs := fixture.requests()
	if last := inputs[len(inputs)-1]; last.Prompt != written || last.NegativePrompt != excluded || last.Enhanced() || last.Layout != nil {
		t.Errorf("the run after the prompt was used asked %+v; want the prompt and negative used, as written", last)
	}
	check(`window.overgo.errors.length===0`)
	webuilane.Leg(t, "moss-prompts-used", "MOSS's generation page shows the prompt and negative prompt each run's picture was drawn from, as the MOSS Ultra workflow does, and offers each back as the manual prompt.")
}
