//go:build windows

package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/processmeasure"
	"overgo/internal/testcheck"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserMOSSRunShowsBesideTheForm holds moss-page/result (the owner's review of 2026-11-24, point 2): at
// 1280 x 800, MOSS's Generate page shows a run beside its form, where the form need not scroll for it. A held run
// that has finished 3 of its 8 sampling steps shows "step 3 of 8" and its progress bar, in view and to the right of
// the form; the finished run's picture is in view as it ends; and Run stays in view with the form scrolled to its top
// or its foot. The page had shown a finished picture below the whole form, off screen at this size, and no step of a
// picture's run, as no sampler reported one.
func TestWebUIBrowserMOSSRunShowsBesideTheForm(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the run-beside leg uses Chromium through cmd/webui-lane")
	release := make(chan struct{})
	fixture := newMOSSGenerationFixture(t, func(ctx context.Context, _ artifact.Repository, _ modelrecipe.CapabilityEvidenceSelection, _ latentimage.Request) error {
		processmeasure.Stepped(ctx, 3, 8)
		select {
		case <-release:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	})
	registered := serveTestRequest(fixture.handler, http.MethodPost, "/library/register", `{"kind":"model","path":`+string(testcheck.Must(json.Marshal(fixture.root))(t))+`}`)
	if registered.Code != http.StatusOK {
		t.Fatalf("register: %d %s", registered.Code, registered.Body)
	}
	server := httptest.NewServer(fixture.handler)
	defer server.Close()
	defer fixture.handler.Close()
	ctx, browser, check, settle := openBrowserLeg(t, server.URL+"/#generation")
	defer browser.Close()
	testcheck.NoError(t, browser.SetViewport(ctx, 1280, 800))
	settle(`!!document.querySelector('#panel-generation.active .workflow-task + .control-grid > label.control')`)
	check(`(() => {
  const panel=document.querySelector('#panel-generation');
  const inView=(node)=>{const box=node.getBoundingClientRect(); return box.width>0 && box.height>0 && box.top>=0 && box.left>=0 && box.bottom<=innerHeight && box.right<=innerWidth;};
  window.bench={
    panel, inView,
    run:()=>[...panel.querySelectorAll('button')].find(button=>button.textContent==='Run'),
    form:()=>panel.querySelector('.workflow-form'),
    result:()=>panel.querySelector('.workflow-result'),
    scrolled:(where)=>{const wrap=panel.closest('.wrap'); wrap.scrollTop=where==='foot'?wrap.scrollHeight:0; return true;},
  };
  const prompt=[...panel.querySelectorAll('label.control')].find(label=>label.querySelector('span').textContent==='prompt').querySelector('textarea,input');
  prompt.value='a lighthouse at dusk';
  prompt.dispatchEvent(new Event('input',{bubbles:true}));
  for (const where of ['top','foot']) {
    bench.scrolled(where);
    const wrap=panel.closest('.wrap'), actions=bench.run().parentElement;
    if (!inView(bench.run())) throw new Error('Run is out of view with the form scrolled to its '+where+': '+JSON.stringify({run:bench.run().getBoundingClientRect(), wrap:wrap.getBoundingClientRect(), height:innerHeight, scroll:[wrap.scrollTop, wrap.scrollHeight, wrap.clientHeight], position:getComputedStyle(actions).position, overflow:[...function*(){for(let node=actions.parentElement;node;node=node.parentElement) yield node.className+':'+getComputedStyle(node).overflow;}()]}));
  }
  bench.scrolled('top');
  bench.run().click();
  return true;
})()`)
	// The held run's steps show beside the form.
	settle(`(() => { const shown=bench.panel.querySelector('.workflow-progress'); return !shown.hidden && shown.value===3 && shown.max===8 && bench.panel.textContent.includes('step 3 of 8'); })()`)
	check(`(() => {
  const progress=bench.panel.querySelector('.workflow-progress');
  if (!bench.inView(progress)) throw new Error('the run\'s progress is out of view: '+JSON.stringify(progress.getBoundingClientRect()));
  if (bench.result().getBoundingClientRect().left < bench.form().getBoundingClientRect().right) throw new Error('the run stands below the form, not beside it');
  return true;
})()`)
	close(release)
	settle(`(() => { const picture=bench.panel.querySelector('.workbench-output'); return !!picture && picture.complete && picture.naturalWidth>0 && !bench.run().disabled; })()`)
	check(`(() => {
  const picture=bench.panel.querySelector('.workbench-output');
  if (!bench.inView(picture)) throw new Error('the run\'s picture is out of view: '+JSON.stringify(picture.getBoundingClientRect()));
  for (const where of ['foot','top']) { bench.scrolled(where); if (!bench.inView(bench.run())) throw new Error('Run is out of view after the run with the form scrolled to its '+where); }
  return window.overgo.errors.length===0;
})()`)
	webuilane.Leg(t, "moss-run-beside", "At 1280 x 800, MOSS's Generate page shows a held run's step and progress beside the form, its finished picture in view, and Run in view with the form scrolled to its top or foot.")
}
