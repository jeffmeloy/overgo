//go:build windows

package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/media"
	"overgo/internal/testcheck"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserMOSSControlsReadAsTheyAreSet holds moss-page/faults (the owner's review of 2026-11-24, point 5):
// MOSS's controls read as they are set.
//   - The canvas's aspect is a one-line choice of ratios.
//   - The schedule's scheduler is named so, not by its field's name.
//   - No choice reads "unset" where the model's default applies.
//   - No control is named Preset twice, the saved settings named apart.
//   - Preview depth map is offered only while a depth picture is chosen and its control shown.
//
// The page had shown the aspect as a box of prose and the scheduler as "name", read "unset", named the section and the
// saved settings both Preset, and offered the preview with no depth picture.
func TestWebUIBrowserMOSSControlsReadAsTheyAreSet(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the control-faults leg uses Chromium through cmd/webui-lane")
	fixture := newMOSSGenerationFixture(t, nil)
	registered := serveTestRequest(fixture.handler, http.MethodPost, "/library/register", `{"kind":"model","path":`+string(testcheck.Must(json.Marshal(fixture.root))(t))+`}`)
	if registered.Code != http.StatusOK {
		t.Fatalf("register: %d %s", registered.Code, registered.Body)
	}
	started := tinyPNG(t).Data
	picture := artifact.Content{Descriptor: artifact.Descriptor{ID: testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, started))(t), Size: uint64(len(started)), MediaType: media.PNGMediaType}, Data: started}
	testcheck.Must(artifact.Publish(t.Context(), fixture.store, artifact.Batch{Key: "fixture/depth-picture", Artifacts: []artifact.Descriptor{picture.Descriptor}, Contents: []artifact.Content{picture}}))(t)
	server := httptest.NewServer(fixture.handler)
	defer server.Close()
	defer fixture.handler.Close()
	_, browser, check, settle := openBrowserLeg(t, server.URL+"/#generation")
	defer browser.Close()
	settle(`!!document.querySelector('#panel-generation.active fieldset[data-control="control"]') && [...document.querySelectorAll('#panel-generation button')].some(button=>button.textContent==='Preview depth map')`)
	check(`(() => {
  const panel=document.querySelector('#panel-generation');
  const named=(name, within)=>[...within.querySelectorAll('label.control')].find(label=>label.querySelector('span').textContent===name);
  const group=(name)=>panel.querySelector('fieldset[data-control="'+name+'"]');
  const aspect=named('aspect', group('canvas'));
  if (!aspect || aspect.querySelector('textarea') || !aspect.querySelector('select') || ![...aspect.querySelectorAll('option')].some(option=>option.value==='16:9')) throw new Error('the canvas\'s aspect is not a one-line choice of ratios: '+(aspect && aspect.innerHTML));
  if (!named('scheduler', group('schedule')) || named('name', group('schedule'))) throw new Error('the schedule names its scheduler "name"');
  const unset=[...panel.querySelectorAll('option')].filter(option=>option.textContent==='unset');
  if (unset.length) throw new Error(unset.length+' choices read unset');
  const presets=[...panel.querySelectorAll('label.control > span, [aria-label]')].filter(node=>(node.getAttribute('aria-label')||node.textContent).trim().toLowerCase()==='preset');
  if (presets.length>1) throw new Error(presets.length+' controls are named preset');
  if (!panel.querySelector('select[aria-label="saved settings"]')) throw new Error('the saved settings are not named so');
  window.bench={panel, group,
    preview:()=>[...panel.querySelectorAll('button')].find(button=>button.textContent==='Preview depth map'),
    offered:()=>{const shown=bench.preview(); return !!shown && !shown.closest('[hidden]');},
    picture:()=>named('picture', group('control')).querySelector('input[type=hidden]'),
    section:(value)=>{const select=panel.querySelector('.workflow-task + .control-grid').firstElementChild.querySelector('select'); select.value=value; select.dispatchEvent(new Event('change',{bubbles:true}));},
  };
  if (bench.offered()) throw new Error('Preview depth map is offered with no depth picture');
  bench.section('ultra-depth-control');
  bench.picture().value=` + strconv.Quote(picture.Descriptor.ID.String()) + `;
  bench.picture().dispatchEvent(new Event('change',{bubbles:true}));
  const control=group('control');
  if (!bench.offered()) throw new Error('Preview depth map is not offered with a depth picture: '+JSON.stringify({value:control.value, hidden:!!control.closest('[hidden]'), host:bench.preview() && bench.preview().closest('[hidden]') && bench.preview().closest('[hidden]').outerHTML.slice(0,120)}));
  bench.section('ultra-text-to-image');
  if (bench.offered()) throw new Error('Preview depth map is offered under a section that runs no depth control');
  return window.overgo.errors.length===0;
})()`)
	webuilane.Leg(t, "moss-control-faults", "MOSS's aspect is a one-line choice, its scheduler named so, no choice reads unset, one control at most is named preset, and Preview depth map shows only with a depth picture under a section that runs it.")
}
