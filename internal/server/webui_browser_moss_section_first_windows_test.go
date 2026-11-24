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

// TestWebUIBrowserMOSSSectionIsTheFirstChoice holds moss-page/sections (the owner's review of 2026-11-24, point 1): the
// Generate page's first control for MOSS is the MOSS Ultra workflow's section, outside Advanced options, offering the
// six sections first in the workflow's order and every choice by its plain name ("Depth control (Turbo)"), its value
// the preset the request names; a section chosen by its name runs as that preset. The sections had been the preset
// control's ids (ultra-depth-control-turbo) behind Advanced options.
func TestWebUIBrowserMOSSSectionIsTheFirstChoice(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the section-first leg uses Chromium through cmd/webui-lane")
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
	settle(`!!document.querySelector('#panel-generation.active .workflow-task + .control-grid > label.control')`)
	check(`(() => {
  const panel=document.querySelector('#panel-generation');
  const first=panel.querySelector('.workflow-task + .control-grid').firstElementChild;
  const named=first.querySelector('span')?.textContent;
  if (named!=='section') throw new Error('the first control is '+JSON.stringify(named)+'; want the section');
  if (first.closest('.advanced-controls')) throw new Error('the section is behind Advanced options');
  const offered=[...first.querySelector('select').options].filter(option=>option.value!=='').map(option=>[option.value,option.text]);
  const want=[['ultra-text-to-image','Text to image'],['ultra-image-to-image','Image to image'],['ultra-depth-control','Depth control'],
    ['ultra-text-to-image-turbo','Text to image (Turbo)'],['ultra-image-to-image-turbo','Image to image (Turbo)'],['ultra-depth-control-turbo','Depth control (Turbo)']];
  if (JSON.stringify(offered.slice(0,want.length))!==JSON.stringify(want)) throw new Error('the section offers first '+JSON.stringify(offered));
  const written=offered.filter(([value,text])=>value===text);
  if (written.length) throw new Error('choices offered by their ids: '+JSON.stringify(written));
  if ([...panel.querySelectorAll('label.control > span')].some(span=>span.textContent==='preset')) throw new Error('a preset control is offered beside the section');
  const select=first.querySelector('select');
  select.value=[...select.options].find(option=>option.text==='Text to image').value;
  select.dispatchEvent(new Event('change',{bubbles:true}));
  const prompt=[...panel.querySelectorAll('label.control')].find(label=>label.querySelector('span').textContent==='prompt').querySelector('textarea,input');
  prompt.value='a lighthouse at dusk';
  prompt.dispatchEvent(new Event('input',{bubbles:true}));
  window.runButton=[...panel.querySelectorAll('button')].find(button=>button.textContent==='Run');
  window.runButton.click();
  return true;
})()`)
	settle(`!!document.querySelector('#panel-generation .workbench-output') && !window.runButton.disabled`)
	inputs := fixture.requests()
	if len(inputs) != 1 || inputs[0].Preset != "ultra-text-to-image" || inputs[0].Prompt != "a lighthouse at dusk" {
		t.Fatalf("the section chosen by its name ran %+v; want one request naming the preset ultra-text-to-image", inputs)
	}
	check(`window.overgo.errors.length===0`)
	webuilane.Leg(t, "moss-section-first", "MOSS's Generate page leads with the MOSS Ultra workflow's section, its six sections first in the workflow's order, every choice by its plain name, and a section chosen by its name runs as its preset.")
}
