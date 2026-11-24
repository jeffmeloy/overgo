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

// TestWebUIBrowserMOSSCompareIsAudited holds moss-page/audit (the owner's review of 2026-11-24, point 6): a finished
// image to image run's A/B compare, the state no leg had captured, is held to the layout audit at the desktop and phone
// sizes, its divide a finger's control. A stat the page was not given a value for reads "not recorded", as Runtime's
// requested residency had read "undefined". The divide had been the browser's 16 px slider.
func TestWebUIBrowserMOSSCompareIsAudited(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the compare-audit leg uses Chromium through cmd/webui-lane")
	fixture := newMOSSGenerationFixture(t, nil)
	registered := serveTestRequest(fixture.handler, http.MethodPost, "/library/register", `{"kind":"model","path":`+string(testcheck.Must(json.Marshal(fixture.root))(t))+`}`)
	if registered.Code != http.StatusOK {
		t.Fatalf("register: %d %s", registered.Code, registered.Body)
	}
	started := tinyPNG(t).Data
	picture := artifact.Content{Descriptor: artifact.Descriptor{ID: testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, started))(t), Size: uint64(len(started)), MediaType: media.PNGMediaType}, Data: started}
	testcheck.Must(artifact.Publish(t.Context(), fixture.store, artifact.Batch{Key: "fixture/compare-picture", Artifacts: []artifact.Descriptor{picture.Descriptor}, Contents: []artifact.Content{picture}}))(t)
	server := httptest.NewServer(fixture.handler)
	defer server.Close()
	defer fixture.handler.Close()
	ctx, browser, check, settle := openBrowserLeg(t, server.URL+"/#generation")
	defer browser.Close()
	settle(`!!document.querySelector('#panel-generation.active .workflow-task + .control-grid > label.control')`)
	check(`(() => {
  const panel=document.querySelector('#panel-generation');
  const section=panel.querySelector('.workflow-task + .control-grid').firstElementChild.querySelector('select');
  section.value='ultra-image-to-image'; section.dispatchEvent(new Event('change',{bubbles:true}));
  const source=[...panel.querySelectorAll('label.control')].find(label=>label.querySelector('span').textContent==='picture' && !label.closest('fieldset[data-control]')).querySelector('input');
  source.value=` + strconv.Quote(picture.Descriptor.ID.String()) + `; source.dispatchEvent(new Event('change',{bubbles:true}));
  const prompt=[...panel.querySelectorAll('label.control')].find(label=>label.querySelector('span').textContent==='prompt').querySelector('textarea');
  prompt.value='a harbour at noon'; prompt.dispatchEvent(new Event('input',{bubbles:true}));
  window.runButton=[...panel.querySelectorAll('button')].find(button=>button.textContent==='Run');
  window.runButton.click();
  return true;
})()`)
	settle(`(() => { const frame=document.querySelector('#panel-generation .ab-frame'); return !!frame && [...frame.querySelectorAll('img')].every(shown=>shown.complete && shown.naturalWidth>0) && !window.runButton.disabled; })()`)
	check(`(() => {
  const divide=document.querySelector('#panel-generation .ab-divide').getBoundingClientRect();
  if (divide.height<24) throw new Error('the A/B divide is '+divide.height+' px tall; a finger needs 24');
  const shown=window.overgo.stat('Requested residency', undefined).textContent;
  if (shown.includes('undefined') || !shown.includes('not recorded')) throw new Error('a stat with no value reads '+JSON.stringify(shown));
  return window.overgo.errors.length===0;
})()`)
	for _, viewport := range webuilane.ScreenViewports {
		testcheck.NoError(t, webuilane.SettleViewport(ctx, browser, viewport))
		for _, finding := range testcheck.Must(browser.LayoutAudit(ctx))(t) {
			t.Errorf("%s: the compare state's layout: %s", viewport.Name, finding)
		}
	}
	webuilane.Leg(t, "moss-compare-audit", "A finished image to image run's A/B compare on MOSS's Generate page passes the layout audit at the desktop and phone sizes, its divide a full-size control, and a stat with no value reads not recorded.")
}
