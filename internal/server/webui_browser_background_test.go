package server

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"sync/atomic"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
)

func TestWebUIBrowserColdActivity(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.ShortIntegration + ": cold activity uses Chromium")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	idle := &IdleShell{Repository: store, Catalog: func(context.Context) ([]discovery.CatalogEntry, bool, error) { return nil, false, nil }, Intake: LibraryIntake{
		ListProviderModels: func(context.Context, string, string) ([]ProviderModel, error) {
			return []ProviderModel{{ID: "fixture/cold-model", ContextLength: 4096}}, nil
		},
		DeclareProvider: laneIdleIntake().DeclareProvider,
	}}
	front := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Overgo-Swap-Proxy", "")
		idle.ServeHTTP(w, r)
	}))
	defer front.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	browser, err := webuilane.Open(ctx, path, front.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	if err := browser.Eventually(ctx, `!!document.querySelector('#cold-start')`); err != nil {
		t.Fatal(err)
	}
	if err := browser.Eventually(ctx, `document.querySelector('#model-pill').textContent==='Choose a model' && !document.querySelector('#panel-chat.active') && document.querySelector('#proxy-dot').classList.contains('ok') && overgo.errors.length===0`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `(() => {window.coldActivityIdle=false;const stop=overgo.runtimeEvents.subscribe(name=>{if(name==='stream.idle'){coldActivityIdle=true;stop();}});return true;})()`)
	if err := browser.Eventually(ctx, `coldActivityIdle`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `(() => {[...document.querySelectorAll('#cold-start button')].find(button=>button.textContent==='Open the Library').click();return true;})()`)
	if err := browser.Eventually(ctx, `!!document.querySelector('#panel-library.active input[aria-label="provider name"]')`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `overgo.errors.length===0 && document.querySelector('#global-operation-shell').hidden`)
	assertBrowserPredicate(t, ctx, browser, `(() => {const field=name=>document.querySelector('#panel-library input[aria-label="provider '+name+'"]');field('name').value='cold';field('endpoint').value='http://localhost';field('key variable').value='COLD_FIXTURE_KEY';[...document.querySelectorAll('#panel-library button')].find(button=>button.textContent==="list the provider's models").click();return true;})()`)
	if err := browser.Eventually(ctx, `[...document.querySelectorAll('#panel-library button')].some(button=>button.textContent==='fixture/cold-model · 4096')`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `(() => {[...document.querySelectorAll('#panel-library button')].find(button=>button.textContent==='fixture/cold-model · 4096').click();[...document.querySelectorAll('#panel-library button')].find(button=>button.textContent==='declare a hosted provider').click();return true;})()`)
	if err := browser.Eventually(ctx, `document.querySelector('#panel-library').textContent.includes('remote://cold/fixture/cold-model') || document.querySelector('#panel-library').textContent.includes('worktree is dirty')`); err != nil {
		var panel string
		_ = browser.Evaluate(t.Context(), `document.querySelector('#panel-library').innerText`, &panel)
		t.Fatalf("%v: %s", err, panel)
	}
}

func TestWebUIBrowserBackgroundWork(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.ShortIntegration + ": background work runs through cmd/webui-lane")
	}
	root := t.TempDir()
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	handler := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{pieces: []string{"Background answer"}}))
	handler.config.APIKey = testAPIKey
	var registrations atomic.Int32
	registeredRecipe := testutil.ArtifactID(t, artifact.KindRecipe, "background registered recipe")
	handler.config.LibraryIntake = LibraryIntake{
		ModelFiles: func(path, projector string) (string, string, error) { return path, projector, nil },
		Register: func(ctx context.Context, repository *overgodb.Store, path, _ string) (map[string]any, error) {
			if registrations.Add(1) == 1 {
				return nil, errors.New("Controlled registration failure")
			}
			_, err := artifact.CommitBatch(ctx, repository, artifact.Batch{Key: "background/library", Artifacts: []artifact.Descriptor{{ID: registeredRecipe}}})
			return map[string]any{"recipe": registeredRecipe.String(), "path": path}, err
		},
	}
	defer handler.Close()
	server := httptest.NewServer(handler)
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	browser, err := webuilane.Open(ctx, path, server.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	check := func(expression string) { t.Helper(); assertBrowserPredicate(t, ctx, browser, expression) }
	settle := func(expression string) {
		t.Helper()
		if err := browser.Eventually(ctx, expression); err != nil {
			t.Fatalf("%s: %v", expression, err)
		}
	}
	settle(`!!document.querySelector('.composer textarea')`)
	check(`(() => {
  window.backgroundProbe={};window.backgroundFetch=fetch;window.backgroundAuth=[];
  window.fetch=function(path,options){if(path==='/runtime/activity/stream')backgroundAuth.push(options.headers.Authorization||'');return backgroundFetch.apply(this,arguments);};
  window.backgroundComposer=document.querySelector('.composer');
  document.getElementById('settings-toggle').click();const key=document.getElementById('api-key');key.value=` + strconv.Quote(testAPIKey) + `;key.dispatchEvent(new Event('change'));
  backgroundProbe.auth_restart_uses_new_key=backgroundAuth[0]==='Bearer '+key.value;window.fetch=backgroundFetch;
  document.getElementById('settings-dialog').close();return true;
})()`)
	settle(`!!document.querySelector('.composer') && document.querySelector('.composer')!==backgroundComposer`)
	// A replaced connection may report its failure after the new snapshot.
	check(`(async () => {
  window.backgroundEvents=overgo.api.events;let calls=0;
  overgo.api.events=async function(path,publish,options){if(++calls===1)return new Promise((resolve,reject)=>window.backgroundRejectOld=reject);publish('operation.snapshot',[]);return new Promise(()=>{});};
  overgo.runtimeEvents.restart();overgo.runtimeEvents.restart();backgroundRejectOld(new Error('Obsolete connection failure'));
  await new Promise(requestAnimationFrame);
  let stale=false;const unsubscribe=overgo.runtimeEvents.subscribe((name,value)=>{if(name==='stream.error'&&value.message==='Obsolete connection failure')stale=true;});
  await new Promise(requestAnimationFrame);unsubscribe();backgroundProbe.replaced_stream_error_is_ignored=!stale;
  overgo.api.events=backgroundEvents;overgo.runtimeEvents.restart();return true;
})()`)
	settle(`document.getElementById('global-operation-shell').hidden`)
	check(`(() => {document.querySelector('[data-draft-setting=tokens]').value='0';return true;})()`)
	say(t, ctx, browser, "Keep this invalid-settings draft")
	check(`(() => {
  backgroundProbe.invalid_settings_are_reachable=document.getElementById('settings-dialog').open && document.querySelector('.composer textarea').value==='Keep this invalid-settings draft';
  document.querySelector('[data-draft-setting=tokens]').value='2';document.getElementById('settings-dialog').close();return true;
})()`)
	say(t, ctx, browser, "Original editable work")
	settle(`!!document.querySelector('.msg.assistant .body') && !document.querySelector('.send-button').disabled`)
	check(`(() => {[...document.querySelectorAll('.turn-action')].find(button=>button.textContent==='Edit and resend').click();return true;})()`)
	settle(`!!document.querySelector('.turn-editor textarea')`)
	check(`(() => {
  document.querySelector('.turn-editor textarea').value='Keep this edited work';document.querySelector('[data-draft-setting=tokens]').value='0';document.querySelector('.turn-editor button[type=submit]').click();
  backgroundProbe.invalid_settings_preserve_editor=!!document.querySelector('.turn-editor textarea') && document.querySelector('.turn-editor textarea').value==='Keep this edited work';
  document.querySelector('[data-draft-setting=tokens]').value='2';document.getElementById('settings-dialog').close();return true;
})()`)
	check(`(() => {
  window.backgroundStream=overgo.api.stream;window.backgroundRequests=0;
  overgo.api.stream=async function(path,body,options){if(path==='/v1/responses'){backgroundRequests++;const response=await backgroundStream.call(this,path,body,options);await response.body.cancel();throw new TypeError('Connection lost before delivering the response ID');}return backgroundStream.call(this,path,body,options);};return true;
})()`)
	say(t, ctx, browser, "Uncertain work")
	settle(`document.querySelector('#panel-chat').textContent.includes('outcome is unknown')`)
	check(`(() => {backgroundProbe.unknown_outcome_blocks_resend=document.querySelector('.send-button').disabled;document.querySelector('.send-button').click();return true;})()`)
	check(`(async () => {await new Promise(requestAnimationFrame);backgroundProbe.unknown_outcome_has_one_request=backgroundRequests===1;overgo.api.stream=backgroundStream;return true;})()`)
	check(`(() => {window.backgroundComposer=document.querySelector('.composer');overgo.openConversation(overgo.conversation());return true;})()`)
	settle(`!!document.querySelector('.composer') && document.querySelector('.composer')!==backgroundComposer`)
	check(`(() => {backgroundProbe.unknown_outcome_survives_remount=document.querySelector('.send-button').disabled && document.querySelector('.composer-extras').textContent.includes('Uncertain work');return true;})()`)
	check(`(() => {
  [...document.querySelectorAll('.composer-extras button')].find(button=>button.textContent==='Allow sending again').click();
  overgo.capabilities().modes.push({id:'image',label:'Image fixture',enabled:true});window.backgroundGet=overgo.api.get;window.backgroundPost=overgo.api.post;
  const backgroundCaps=structuredClone(overgo.capabilities());
  overgo.api.get=async function(path,options){if(path==='/generation/capabilities')return [{task:'image',recipe:'background-image-fixture',name:'Image fixture',controls:[{name:'source',label:'Source file',type:'artifact',required:true}]}];const result=await backgroundGet.call(this,path,options);if(path==='/workspace/manifest')result.model=structuredClone(backgroundCaps);return result;};
  window.backgroundComposer=document.querySelector('.composer');overgo.openConversation(null);return true;
})()`)
	settle(`!!document.querySelector('.composer') && document.querySelector('.composer')!==backgroundComposer`)
	check(`(() => {const mode=document.querySelector('.composer select[aria-label=mode]');mode.value='image';mode.dispatchEvent(new Event('change'));return true;})()`)
	settle(`!!document.querySelector('.mode-controls label.control input')`)
	say(t, ctx, browser, "Keep the source-required prompt")
	check(`(async () => {await new Promise(requestAnimationFrame);backgroundProbe.required_input_preserves_prompt=document.querySelector('.composer textarea').value==='Keep the source-required prompt';return true;})()`)
	entered := make(chan context.Context, 1)
	runID := testutil.ArtifactID(t, artifact.KindRun, "background cancelled run")
	operationID, err := handler.operations.Submit(ctx, operation.Request{Task: recipe.TaskGeneration, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "background operation fixture")}, func(ctx context.Context, reporter operation.Reporter) (operation.Completion, error) {
		entered <- ctx
		<-ctx.Done()
		return operation.Completion{Run: runID}, ctx.Err()
	})
	if err != nil {
		t.Fatal(err)
	}
	defer handler.operations.Cancel(operationID)
	select {
	case <-entered:
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	check(`(() => {
  window.backgroundAccepted=false;
  overgo.api.post=async function(path,body,options){if(path==='/generation/run'){backgroundAccepted=true;return {operation:` + strconv.Quote(operationID.String()) + `};}return backgroundPost.call(this,path,body,options);};
  document.querySelector('.mode-controls label.control input').value='fixture-file';return true;
})()`)
	say(t, ctx, browser, "Cancel the actual operation")
	settle(`backgroundAccepted && !document.querySelector('.stop-button').hidden`)
	check(`(() => {document.querySelector('.stop-button').click();return true;})()`)
	settle(`!document.querySelector('.send-button').disabled`)
	if status, found := handler.operations.Status(operationID); !found || status.State != operation.StateCancelled {
		t.Errorf("background recovery: Stop left the operation in state %s: %s", status.State, status.Failure)
	}
	// Stop queued before admission must cancel once the real receipt arrives.
	running, blocked := publishBrowserLaneOperations(t, handler)
	defer handler.operations.Cancel(running)
	defer handler.operations.Cancel(blocked)
	check(`(() => {
  window.backgroundCancelCount=0;window.backgroundReleaseAdmission=null;
  overgo.api.post=async function(path,body,options){
    if(path==='/generation/run'){await new Promise(resolve=>window.backgroundReleaseAdmission=resolve);return {operation:` + strconv.Quote(running.String()) + `};}
    if(path==='/operations/cancel' && ++backgroundCancelCount===1)throw new Error('Controlled cancellation failure');
    return backgroundPost.call(this,path,body,options);
  };return true;
})()`)
	say(t, ctx, browser, "Stop before receipt")
	settle(`!!backgroundReleaseAdmission`)
	check(`(() => {document.querySelector('.stop-button').click();backgroundProbe.early_stop_waits_for_receipt=backgroundCancelCount===0 && document.querySelector('.stop-button').disabled; backgroundReleaseAdmission();return true;})()`)
	settle(`document.querySelector('#panel-chat').textContent.includes('Controlled cancellation failure') && !document.querySelector('.stop-button').disabled`)
	check(`(() => {document.querySelector('.stop-button').click();return true;})()`)
	settle(`!document.querySelector('.send-button').disabled`)
	waitBrowserOperationState(t, handler, running, operation.StateCancelled)
	check(`backgroundCancelCount===2 && document.querySelector('.composer textarea').value==='Stop before receipt'`)
	// A different operation completes while the user navigates and composes.
	finishWork := make(chan struct{})
	outputBytes := []byte("Completed background output")
	outputID, err := artifact.IdentifyBytes(artifact.KindFile, outputBytes)
	if err != nil {
		t.Fatal(err)
	}
	work, err := handler.operations.Submit(ctx, operation.Request{Task: recipe.TaskGeneration, Recipe: registeredRecipe}, func(ctx context.Context, _ operation.Reporter) (operation.Completion, error) {
		select {
		case <-finishWork:
			return operation.Completion{Run: runID, Outputs: []artifact.ID{outputID}}, nil
		case <-ctx.Done():
			return operation.Completion{Run: runID}, ctx.Err()
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	defer handler.operations.Cancel(work)
	check(`(() => {overgo.api.post=async function(path,body,options){if(path==='/generation/run')return {operation:` + strconv.Quote(work.String()) + `};return backgroundPost.call(this,path,body,options);};return true;})()`)
	say(t, ctx, browser, "Continue in background")
	settle(`!![...document.querySelectorAll('#panel-chat button')].find(button=>button.textContent==='View activity') && !document.querySelector('.stop-button').hidden`)
	check(`(() => {window.backgroundDraft=document.querySelector('.composer textarea');backgroundDraft.value='Draft written during work';backgroundDraft.dispatchEvent(new Event('input'));location.hash='library';return true;})()`)
	settle(`!!document.querySelector('#panel-library.active input[placeholder="model GGUF or directory on disk"]')`)
	check(`(() => {
  const panel=document.querySelector('#panel-library');panel.querySelector('input[placeholder="model GGUF or directory on disk"]').value='fixture.gguf';
  [...panel.querySelectorAll('button')].find(button=>button.textContent==='register a local model').click();
  [...panel.querySelectorAll('button')].find(button=>button.textContent==='register').click();return true;
})()`)
	settle(`document.querySelector('#panel-library').textContent.includes('Controlled registration failure')`)
	check(`(() => {
  window.backgroundRegisterRelease=null;window.backgroundRegisterRequests=0;
  overgo.api.post=async function(path,body,options){if(path==='/library/register'){backgroundRegisterRequests++;await new Promise(resolve=>window.backgroundRegisterRelease=resolve);}return backgroundPost.call(this,path,body,options);};
  const register=[...document.querySelectorAll('#panel-library button')].find(button=>button.textContent==='register');register.click();register.click();return true;
})()`)
	settle(`!!backgroundRegisterRelease`)
	check(`(() => {backgroundProbe.registration_not_duplicated=backgroundRegisterRequests===1;location.hash='chat';return true;})()`)
	settle(`!!document.querySelector('#panel-chat.active .composer textarea')`)
	check(`(() => {backgroundProbe.navigation_preserves_live_draft=document.querySelector('.composer textarea').value==='Draft written during work' && !document.querySelector('.activity-dialog').open;backgroundRegisterRelease();return true;})()`)
	settle(`!![...document.querySelectorAll('#panel-library button')].find(button=>button.textContent==='registered')`)
	if registrations.Load() != 2 {
		t.Fatalf("registration attempts=%d, want failure + one retry", registrations.Load())
	}
	check(`(() => {location.hash='library';return true;})()`)
	settle(`!!document.querySelector('#panel-library.active')`)
	check(`(async () => {
  let calls=0,releaseOld;overgo.api.get=async function(path,options){if(path==='/catalog/models' && ++calls===1)return new Promise(resolve=>releaseOld=resolve);return backgroundGet.call(this,path,options);};
  const refresh=[...document.querySelectorAll('#panel-library button')].find(button=>button.textContent==='Refresh catalog');refresh.click();refresh.click();await new Promise(requestAnimationFrame);
  releaseOld({models:[{model:'obsolete',location:'obsolete-catalog.gguf',present:true}]});await new Promise(requestAnimationFrame);
  backgroundProbe.catalog_refresh_keeps_current_form=!document.querySelector('#panel-library').textContent.includes('obsolete-catalog') && document.querySelector('#panel-library input[placeholder="model GGUF or directory on disk"]').value==='fixture.gguf';
  overgo.api.get=backgroundGet;location.hash='chat';return true;
})()`)
	settle(`!!document.querySelector('#panel-chat.active .composer')`)
	external, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	_, err = artifact.CommitBatch(ctx, external, artifact.Batch{Key: "background/external", Contents: []artifact.Content{{Descriptor: artifact.Descriptor{ID: outputID, Size: uint64(len(outputBytes)), MediaType: "text/plain"}, Data: outputBytes}}})
	if err != nil {
		external.Close()
		t.Fatal(err)
	}
	head, _ := external.Head()
	if err := external.Close(); err != nil {
		t.Fatal(err)
	}
	check(`(async () => {const view=await backgroundGet('/store/deltas');return view.head===` + strconv.Quote(head.String()) + ` && document.querySelector('.composer textarea').value==='Draft written during work';})()`)
	close(finishWork)
	settle(`!document.querySelector('.send-button').disabled`)
	waitBrowserOperationState(t, handler, work, operation.StateCompleted)
	check(`(() => {overgo.showOperation(` + strconv.Quote(work.String()) + `);return true;})()`)
	settle(`document.querySelector('.operation-detail').textContent.includes('Download result')`)
	check(`!![...document.querySelectorAll('.operation-detail a')].find(link=>new URL(link.href).searchParams.get('id')===` + strconv.Quote(outputID.String()) + `)`)
	check(`(() => {
  window.backgroundAnchorClick=HTMLAnchorElement.prototype.click;window.backgroundBlob=overgo.api.blob;window.backgroundOutputRead='';window.backgroundOutputDownload=false;
  HTMLAnchorElement.prototype.click=function(){if(this.href.startsWith('blob:')){backgroundOutputDownload=true;return;}return backgroundAnchorClick.call(this);};
  overgo.api.blob=async function(path,options){const body=await backgroundBlob.call(this,path,options);backgroundOutputRead=await body.text();return body;};
  [...document.querySelectorAll('.operation-detail a')].find(link=>new URL(link.href).searchParams.get('id')===` + strconv.Quote(outputID.String()) + `).click();return true;
})()`)
	settle(`backgroundOutputDownload && backgroundOutputRead==='Completed background output'`)
	check(`(() => {HTMLAnchorElement.prototype.click=backgroundAnchorClick;overgo.api.blob=backgroundBlob;return true;})()`)
	// A failed decision remains retryable; the actual receipt supplies completion.
	check(`(() => {overgo.showOperation(` + strconv.Quote(blocked.String()) + `);return true;})()`)
	settle(`!![...document.querySelectorAll('.operation-detail button')].find(button=>button.textContent==='Grant retry-browser')`)
	for _, viewport := range []webuilane.Viewport{{Name: "desktop", Width: 1280, Height: 900}, {Name: "phone", Width: 390, Height: 844}, {Name: "phone-keyboard", Width: 390, Height: 380}, {Name: "small-phone", Width: 320, Height: 568}} {
		for _, scheme := range webuilane.ColourSchemes {
			if err := browser.SetColorScheme(ctx, scheme); err != nil {
				t.Fatal(err)
			}
			if _, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "background-decision-"+scheme); err != nil {
				t.Fatal(err)
			}
			check(`(() => {const dialog=document.querySelector('.activity-dialog'),r=dialog.getBoundingClientRect();return dialog.open && r.left>=0 && r.right<=innerWidth && r.top>=0 && r.bottom<=innerHeight && dialog.scrollWidth<=dialog.clientWidth && document.querySelector('.activity-list').hidden;})()`)
		}
	}
	check(`(() => {
  window.backgroundDecisionCount=0;overgo.api.post=async function(path,body,options){if(path==='/operations/decision' && ++backgroundDecisionCount===1)throw new Error('Controlled decision failure');return backgroundPost.call(this,path,body,options);};
  [...document.querySelectorAll('.operation-detail button')].find(button=>button.textContent==='Grant retry-browser').click();return true;
})()`)
	settle(`document.querySelector('.operation-detail').textContent.includes('Controlled decision failure')`)
	check(`(() => {[...document.querySelectorAll('.operation-detail button')].find(button=>button.textContent==='Retry action').click();return true;})()`)
	settle(`document.querySelector('.operation-detail').textContent.includes('completed')`)
	waitBrowserOperationState(t, handler, blocked, operation.StateCompleted)
	check(`(() => {document.querySelector('[aria-label="Close activity"]').click();overgo.api.post=backgroundPost;overgo.api.get=backgroundGet;return true;})()`)
	// Stale receipt failures cannot replace a newer selected operation.
	check(`(async () => {
  overgo.api.get=async function(path,options){if(path.includes('/operations/evidence?id=late-fixture'))return new Promise((resolve,reject)=>window.backgroundLateReceipt=reject);return backgroundGet.call(this,path,options);};
  overgo.showOperation('late-fixture');overgo.showOperation(` + strconv.Quote(work.String()) + `);backgroundLateReceipt(new Error('Obsolete receipt failure'));await new Promise(requestAnimationFrame);return true;
})()`)
	settle(`document.querySelector('.operation-detail').textContent.includes('Download result')`)
	check(`(() => {backgroundProbe.stale_receipt_is_ignored=!document.querySelector('.operation-detail').textContent.includes('Obsolete receipt');document.querySelector('[aria-label="Close activity"]').click();overgo.api.get=backgroundGet;return true;})()`)
	// Stream fallback keeps a blocked receipt alive and releases requests on abort.
	check(`(async () => {
  let rejectStream,publishStream,waitSignal,waits=0;
  overgo.api.events=async function(path,publish,options){publishStream=publish;return new Promise((resolve,reject)=>{rejectStream=reject;options.signal.addEventListener('abort',()=>reject(new DOMException('aborted','AbortError')),{once:true});});};
  overgo.api.get=async function(path,options){if(path.startsWith('/operations/wait?id=fallback-fixture')){waits++;waitSignal=options.signal;return new Promise((resolve,reject)=>options.signal.addEventListener('abort',()=>reject(new DOMException('aborted','AbortError')),{once:true}));}return backgroundGet.call(this,path,options);};
  overgo.runtimeEvents.restart();const abort=new AbortController();const waiting=overgo.waitOperation('fallback-fixture',null,abort.signal).catch(error=>error.name);
  rejectStream(new Error('Controlled stream interruption'));await new Promise(requestAnimationFrame);abort.abort();backgroundProbe.fallback_aborts=await waiting==='AbortError' && waits===1 && waitSignal.aborted;
  overgo.api.get=async function(path,options){if(path.startsWith('/operations/wait?id=fallback-blocked'))return {id:'fallback-blocked',state:'blocked'};return backgroundGet.call(this,path,options);};
  let finished=false,observed=false;const blocked=overgo.waitOperation('fallback-blocked',status=>{if(status.state==='blocked')observed=true;}).then(status=>{finished=true;return status;});
  await new Promise(requestAnimationFrame);backgroundProbe.blocked_fallback_keeps_waiting=observed && !finished;
  overgo.runtimeEvents.restart();publishStream('operation.snapshot',[{id:'fallback-blocked',task:'fixture',state:'completed'}]);await blocked;
  let connectionError=false;const stop=overgo.runtimeEvents.subscribe(name=>{if(name==='stream.error')connectionError=true;});await new Promise(requestAnimationFrame);stop();backgroundProbe.reconnected_stream_clears_failure=!connectionError;
  overgo.api.events=backgroundEvents;overgo.api.get=backgroundGet;overgo.runtimeEvents.restart();return true;
})()`)
	// Exercise a real structured busy response through the browser API boundary.
	check(`(async () => {
  const before=await backgroundGet('/workspace/manifest');window.backgroundModelBefore=before;window.backgroundModelCurrent=before;window.backgroundBusyAttempts=0;window.backgroundStartupFailures=['resource_busy','insufficient_memory','model_load_failed','unknown_failure'];window.backgroundConversationBefore=JSON.stringify(overgo.conversation());
  const candidate=JSON.parse(JSON.stringify(before));candidate.model.recipe='background-busy-recipe';candidate.model.model='background-busy-model';candidate.model.id='background-busy';window.backgroundModelCandidate=candidate;
  overgo.api.get=async function(path,options){if(path==='/catalog/models')return {models:[{model:'other-model',recipe:'other-recipe',location:'/shared/busy.gguf',present:true},{model:candidate.model.model,recipe:candidate.model.recipe,location:'/shared/busy.gguf',present:true}]};if(path==='/workspace/manifest')return backgroundModelCurrent;return backgroundGet.call(this,path,options);};
  window.fetch=async function(path,options){if(typeof path==='string' && path.startsWith('/health?swap=')){backgroundBusyAttempts++;if(new URL(path,location.origin).searchParams.get('swap')!==candidate.model.model)return new Response(JSON.stringify({error:{message:'Ambiguous or wrong model reference'}}),{status:400});if(backgroundBusyAttempts===4){backgroundModelCurrent=JSON.parse(JSON.stringify(before));backgroundModelCurrent.model.model='wrong-original-artifact-same-recipe';}if(backgroundBusyAttempts<=backgroundStartupFailures.length)return new Response(JSON.stringify({error:{code:backgroundStartupFailures[backgroundBusyAttempts-1],message:'Controlled startup diagnostic'}}),{status:503});backgroundModelCurrent=JSON.parse(JSON.stringify(candidate));if(backgroundBusyAttempts===5)backgroundModelCurrent.model.model='wrong-artifact-same-recipe';return new Response('{}',{headers:{'X-Overgo-Swap-Proxy':candidate.model.model}});}return backgroundFetch.apply(this,arguments);};
  document.querySelector('#model-pill').click();return true;
})()`)
	settle(`!!document.querySelector('[data-serve]')`)
	check(`(() => {const row=[...document.querySelectorAll('dialog[aria-label="Choose a model"] .row')].find(row=>row.querySelector('.picker-facts')?.textContent.includes(backgroundModelCandidate.model.model));if(!row)return false;row.querySelector('[data-serve]').click();return true;})()`)
	for index, message := range []string{"reserved for exclusive work", "not enough free GPU memory", "model could not start", "Controlled startup diagnostic", "server did not confirm the requested model"} {
		t.Logf("model admission response %d: %s", index+1, message)
		settle(`document.querySelector('dialog[aria-label="Choose a model"]').textContent.includes(` + strconv.Quote(message) + `) && [...document.querySelectorAll('dialog[aria-label="Choose a model"] button')].some(button=>button.textContent==='Retry loading model' && !button.disabled)`)
		check(`(() => {
  const picker=document.querySelector('dialog[aria-label="Choose a model"]');
  return document.querySelector('.composer textarea').value==='Draft written during work' && JSON.stringify(overgo.conversation())===backgroundConversationBefore && overgo.capabilities().model===backgroundModelBefore.model.model && !!picker.querySelector('details:not([open])') && backgroundBusyAttempts===` + strconv.Itoa(index+1) + `;
})()`)
		if index >= 3 {
			check(`document.querySelector('.send-button').disabled`)
		} else {
			check(`!document.querySelector('.send-button').disabled`)
		}
		for _, viewport := range []webuilane.Viewport{{Name: "phone", Width: 390, Height: 844}, {Name: "phone-keyboard", Width: 320, Height: 320}} {
			for _, scheme := range webuilane.ColourSchemes {
				if err := browser.SetColorScheme(ctx, scheme); err != nil {
					t.Fatal(err)
				}
				if _, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "model-admission-"+strconv.Itoa(index)+"-"+scheme); err != nil {
					t.Fatal(err)
				}
			}
		}
		check(`(() => {if(backgroundBusyAttempts!==` + strconv.Itoa(index+1) + `)return false;[...document.querySelectorAll('dialog[aria-label="Choose a model"] button')].find(button=>button.textContent==='Retry loading model').click();return true;})()`)
	}
	settle(`!overgo.modelSwitching() && overgo.capabilities().model===backgroundModelCandidate.model.model`)
	check(`(() => {backgroundProbe.startup_explicit_retry_succeeds=backgroundBusyAttempts===6;window.fetch=backgroundFetch;overgo.api.get=backgroundGet;return overgo.errors.length===0;})()`)
	var probes map[string]bool
	if err := browser.Evaluate(ctx, `backgroundProbe`, &probes); err != nil {
		t.Fatal(err)
	}
	for name, passed := range probes {
		if !passed {
			t.Errorf("background recovery: %s failed", name)
		}
	}
	if !t.Failed() {
		t.Log("background work leg: real stored replies and operations; early Stop and cancel retry; Library failure/retry with retained draft; concurrent store publication; durable outputs and decisions; stale receipt and runtime recovery; typed reservation/memory/load refusal and unknown errors; exact artifact confirmation and explicit retry; Chromium desktop and phone layouts. No native image inference or physical-device claim.")
	}
}
