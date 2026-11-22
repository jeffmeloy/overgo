//go:build windows

package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserMOSSPromptMode holds moss-prompt-mode/page: MOSS's generation page offers the MOSS Ultra workflow's
// one prompt mode and shows the chosen mode's fields alone, as the control list declares them (Control.Shows), naming
// no MOSS field itself: text, and a mode not chosen, the prompt and negative and no layout; enhanced the same, its idea
// and exclusions; the builder the layout and negative and no prompt. A field the chosen mode hides is not sent, so a
// builder's run asks its layout and no prompt typed before, and a text run after it asks its prompt and no layout. The
// page had shown prompt mode, enhance and layout side by side and sent whatever was filled.
func TestWebUIBrowserMOSSPromptMode(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the prompt mode leg uses Chromium through cmd/webui-lane")
	fixture := newMOSSGenerationFixture(t, nil)
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
    label:(name)=>[...panel.querySelectorAll('label.control')].find(label=>label.querySelector('span').textContent===name),
    field:(name)=>bench.label(name).querySelector('input,select,textarea'),
    group:(name)=>panel.querySelector('fieldset[data-control="'+name+'"]'),
    button:()=>[...panel.querySelectorAll('button')].find(button=>button.textContent==='Run'),
    set:(field,value)=>{field.value=value;field.dispatchEvent(new Event('input',{bubbles:true}));},
    mode:(value)=>{const mode=bench.field('prompt mode');mode.value=value;mode.dispatchEvent(new Event('change',{bubbles:true}));},
    shows:()=>({prompt:!bench.label('prompt').hidden, negative:!bench.label('negative prompt').hidden, layout:!bench.group('layout').hidden}),
    run:()=>{panel.querySelectorAll('.workbench-output').forEach(shown=>shown.remove());bench.button().click();},
  };
  panel.querySelector('.advanced-controls').open=true;
  return true;
})()`)
	check(`(() => {
  const want={'':[true,true,false], text:[true,true,false], layout:[true,true,false], enhanced:[true,true,false], builder:[false,true,true]};
  for (const [mode, [prompt, negative, layout]] of Object.entries(want)) {
    bench.mode(mode);
    const shown=bench.shows();
    if (shown.prompt!==prompt || shown.negative!==negative || shown.layout!==layout) throw new Error('prompt mode '+JSON.stringify(mode)+' shows '+JSON.stringify(shown));
  }
  return true;
})()`)
	// A builder's run asks its layout and none of the prompt typed before it.
	check(`(() => {
  bench.mode('text');
  bench.set(bench.field('prompt'),'a prompt typed first');
  bench.set(bench.field('width'),'256');
  bench.set(bench.field('height'),'144');
  bench.mode('builder');
  bench.group('layout').value=JSON.stringify({high_level_description:'a fox at dawn',regions:[{x:0.1,y:0.2,w:0.4,h:0.5,desc:'a fox'}]});
  bench.run();
  return true;
})()`)
	settle(`!!bench.panel.querySelector('.workbench-output') && !bench.button().disabled`)
	// A text run after it asks its prompt and none of the layout.
	check(`(() => { bench.mode('text'); bench.run(); return true; })()`)
	settle(`!!bench.panel.querySelector('.workbench-output') && !bench.button().disabled`)
	inputs := fixture.requests()
	if len(inputs) != 2 {
		t.Fatalf("the page ran %d requests; want the builder's and the text's", len(inputs))
	}
	if builder := inputs[0]; builder.PromptMode != "builder" || builder.Layout == nil || builder.Layout.Description != "a fox at dawn" || builder.Prompt != "" {
		t.Errorf("the builder's run asked %+v; want its layout and no prompt", builder)
	}
	if text := inputs[1]; text.PromptMode != "text" || text.Layout != nil || text.Prompt != "a prompt typed first" {
		t.Errorf("the text run asked %+v; want its prompt and no layout", text)
	}
	check(`window.overgo.errors.length===0`)
	webuilane.Leg(t, "moss-prompt-mode", "MOSS's generation page offers the MOSS Ultra workflow's one prompt mode, shows the chosen mode's fields alone as the control list declares them, and sends no field the mode hides.")
}
