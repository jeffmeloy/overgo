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

// TestWebUIBrowserMOSSShowsOnlyTheControlsTheSectionRuns holds moss-page/groups (the owner's review of 2026-11-24,
// point 4): MOSS's Advanced options are grouped, its steps, seed, guidance and schedule as its sampling, and a section
// shows only what it runs. A text to image section shows no picture to start from, no denoise and no depth control.
// An image to image section shows the picture and its denoise. A depth section shows its depth control, in the
// picture's place in the form, and no denoise. No section chosen shows all. The builder prompt mode puts the layout
// where the prompt was. The page had listed about 25 tuning controls in one list, every section's alike.
func TestWebUIBrowserMOSSShowsOnlyTheControlsTheSectionRuns(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the section-controls leg uses Chromium through cmd/webui-lane")
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
  const form=panel.querySelector('.workflow-task + .control-grid');
  const label=(name, within)=>[...(within||panel).querySelectorAll('label.control')].find(found=>found.querySelector('span').textContent===name);
  const shown=(node)=>!!node && !node.closest('[hidden]');
  const group=(name)=>panel.querySelector('.advanced-group[data-group="'+name+'"]');
  const choose=(select, value)=>{select.value=value; select.dispatchEvent(new Event('change',{bubbles:true}));};
  const section=form.firstElementChild.querySelector('select');
  const sampling=group('sampling');
  if (!sampling) throw new Error('Advanced options hold no sampling group');
  for (const name of ['steps','seed','guidance']) if (!label(name, sampling)) throw new Error('the sampling group holds no '+name);
  if (!sampling.querySelector('fieldset[data-control="schedule"]')) throw new Error('the sampling group holds no schedule');
  const picture=[...panel.querySelectorAll('label.control')].find(found=>found.querySelector('span').textContent==='picture' && !found.closest('fieldset'));
  const control=panel.querySelector('fieldset[data-control="control"]');
  const reads=()=>({picture:shown(picture), denoise:shown(label('denoise')), startingGroup:shown(group('starting picture')), control:shown(control)});
  const want={
    '':{picture:true, denoise:true, startingGroup:true, control:true},
    'ultra-text-to-image':{picture:false, denoise:false, startingGroup:false, control:false},
    'ultra-text-to-image-turbo':{picture:false, denoise:false, startingGroup:false, control:false},
    'ultra-image-to-image':{picture:true, denoise:true, startingGroup:true, control:false},
    'ultra-image-to-image-turbo':{picture:true, denoise:true, startingGroup:true, control:false},
    'ultra-depth-control':{picture:false, denoise:false, startingGroup:false, control:true},
    'ultra-depth-control-turbo':{picture:false, denoise:false, startingGroup:false, control:true},
  };
  for (const [value, expected] of Object.entries(want)) {
    choose(section, value);
    const got=reads();
    if (JSON.stringify(got)!==JSON.stringify(expected)) throw new Error('the section '+JSON.stringify(value)+' shows '+JSON.stringify(got)+'; want '+JSON.stringify(expected));
    if (shown(sampling)!==true) throw new Error('the section '+JSON.stringify(value)+' hides the sampling');
  }
  choose(section, 'ultra-depth-control');
  if (control.closest('.advanced-controls') || control.nextElementSibling!==picture) throw new Error('a depth section\'s control is not where the picture was');
  // The builder's layout stands where the prompt was.
  choose(section, '');
  choose(label('prompt mode').querySelector('select'), 'builder');
  const layout=panel.querySelector('fieldset[data-control="layout"]'), prompt=label('prompt');
  if (!shown(layout) || shown(prompt) || layout.closest('.advanced-controls') || layout.nextElementSibling!==prompt) throw new Error('the builder\'s layout is not where the prompt was');
  return window.overgo.errors.length===0;
})()`)
	webuilane.Leg(t, "moss-section-controls", "MOSS's Advanced options are grouped; each section shows only the controls it runs, a depth section's control where the picture was; and the builder's layout stands where the prompt was.")
}
