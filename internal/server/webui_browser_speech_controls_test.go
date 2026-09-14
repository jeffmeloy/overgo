package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http/httptest"
	"os"
	"strconv"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
)

type speechSurfaceGenerator struct {
	*generationWorkspaceGenerator
	secondRecipe, recipeUsed artifact.ID
	retired                  bool
	speech                   struct {
		Voice       string  `json:"voice"`
		MaxFrames   int     `json:"max_frames"`
		Temperature float64 `json:"temperature"`
	}
}

func (generator *speechSurfaceGenerator) WorkflowCapabilities(ctx context.Context, kind WorkflowKind) ([]WorkflowCapability, error) {
	capabilities, err := generator.generationWorkspaceGenerator.WorkflowCapabilities(ctx, kind)
	if err != nil || kind != WorkflowGeneration {
		return capabilities, err
	}
	generator.mu.Lock()
	defer generator.mu.Unlock()
	if !generator.retired {
		second := generator.capability
		second.Recipe = generator.secondRecipe
		second.Name = "Pocket-TTS alternate (UI fixture)"
		capabilities = append(capabilities, second)
	}
	return capabilities, nil
}

func (generator *speechSurfaceGenerator) ExecuteWorkflow(ctx context.Context, kind WorkflowKind, task recipe.Task, recipeID artifact.ID, raw json.RawMessage, reporter operation.Reporter) (operation.Completion, error) {
	generator.mu.Lock()
	generator.recipeUsed = recipeID
	err := json.Unmarshal(raw, &generator.speech)
	generator.mu.Unlock()
	if err != nil {
		return operation.Completion{}, err
	}
	completion, err := generator.generationWorkspaceGenerator.ExecuteWorkflow(ctx, kind, task, recipeID, raw, reporter)
	if err != nil {
		return completion, err
	}
	return completion, errors.New("speech fixture stopped before synthesis")
}

// This probe reviews the current speech form. Its catalog is a UI fixture;
// neither synthesis completion nor model quality is asserted here.
func TestWebUIBrowserSpeechControls(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.ShortIntegration + ": graphical speech form probe")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	generator := &speechSurfaceGenerator{generationWorkspaceGenerator: &generationWorkspaceGenerator{fakeGenerator: &fakeGenerator{}, repository: store, run: testutil.ArtifactID(t, artifact.KindRun, "speech-surface"), capability: WorkflowCapability{
		Task: recipe.TaskSpeech, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "speech-surface"), Name: "Pocket-TTS (UI fixture)",
		Stages: []recipe.Stage{{Node: recipe.Node{ID: "generate", Module: "test.generate"}}},
		Controls: []WorkflowControl{
			{Name: "text", Type: WorkflowControlText, Required: true},
			{Name: "max_frames", Type: WorkflowControlInteger, Required: true},
			{Name: "seed", Type: WorkflowControlInteger, Required: true},
			{Name: "voice", Type: WorkflowControlText, Required: true, Choices: []string{"alba", "marius"}},
			{Name: "temperature", Type: WorkflowControlNumber},
		},
	}}}
	generator.secondRecipe = testutil.ArtifactID(t, artifact.KindRecipe, "speech-surface-alternate")
	handler, err := New(Config{Repository: store}, generator)
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()
	server := httptest.NewServer(handler)
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeoutCause(t.Context(), 60*time.Second, errors.New("speech surface probe did not settle"))
	defer cancel()
	browser, err := webuilane.Open(ctx, path, server.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	settle := func(expression string) {
		t.Helper()
		if err := browser.Eventually(ctx, expression); err != nil {
			t.Fatalf("%s: %v", expression, err)
		}
	}
	check := func(expression string) { t.Helper(); assertBrowserPredicate(t, ctx, browser, expression) }
	settle(`!!document.querySelector('.composer select[aria-label="mode"] option[value="speech"]')`)
	check(`(()=>{const mode=document.querySelector('.composer select[aria-label="mode"]');mode.value='speech';mode.dispatchEvent(new Event('change'));return true;})()`)
	settle(`!!document.querySelector('.mode-controls select[aria-label="voice"]')`)
	check(`(()=>{document.querySelector('.task-model-selector select[aria-label="generation model"]').focus();return true;})()`)
	pressKey(t, ctx, browser, "ArrowDown", 40)
	settle(`document.querySelector('.task-model-selector select[aria-label="generation model"]').value===` + strconv.Quote(generator.secondRecipe.String()))
	check(`(()=>{const input=document.querySelector('.composer textarea');input.value='Read this document aloud.';input.dispatchEvent(new Event('input'));return true;})()`)
	for _, viewport := range append(webuilane.ScreenViewports, webuilane.Viewport{Name: "phone-keyboard", Width: 390, Height: 380}) {
		findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "speech-primary-controls")
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%s: layout findings %v", viewport.Name, findings)
		check(`(()=>{const voice=document.querySelector('.mode-controls select[aria-label="voice"]'),r=voice.getBoundingClientRect();return r.height>0 && r.top>=0 && r.bottom<=innerHeight && document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)===voice;})()`)
		check(`(()=>{const button=document.querySelector('[aria-label="Open speech options"]');button.focus();return document.activeElement===button;})()`)
		pressKey(t, ctx, browser, "Enter", 13)
		settle(`document.querySelector('#settings-dialog').open && !document.querySelector('[aria-label="Speech options"]').hidden`)
		findings, err = webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "speech-options")
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%s options: layout findings %v", viewport.Name, findings)
		check(`(()=>{const button=document.querySelector('#settings-dialog [data-close-dialog]'),r=button.getBoundingClientRect();button.focus();return r.height>0 && r.top>=0 && r.bottom<=innerHeight && document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)===button;})()`)
		pressKey(t, ctx, browser, "Enter", 13)
		settle(`!document.querySelector('#settings-dialog').open`)
		check(`document.activeElement===document.querySelector('[aria-label="Open speech options"]')`)
	}
	check(`(()=>{document.querySelector('.composer .send-button').focus();return true;})()`)
	check(`(()=>{window.speechControl=(name)=>[...document.querySelectorAll('[aria-label="Speech options"] .control')].find(label=>label.querySelector('span')?.textContent===name)?.querySelector('input');return true;})()`)
	pressKey(t, ctx, browser, "Enter", 13)
	settle(`document.querySelector('#settings-dialog').open && document.activeElement===speechControl('max frames')`)
	check(`document.querySelector('.composer textarea').value==='Read this document aloud.'`)
	for _, field := range []struct{ label, value string }{{"max frames", "120"}, {"seed", "7"}, {"temperature", "0.7"}} {
		check(`(()=>{speechControl('` + field.label + `').focus();return true;})()`)
		if err := browser.Call(ctx, "Input.insertText", map[string]any{"text": field.value}, nil); err != nil {
			t.Fatal(err)
		}
	}
	check(`(()=>{document.querySelector('#settings-dialog [data-close-dialog]').focus();return true;})()`)
	pressKey(t, ctx, browser, "Enter", 13)
	check(`(()=>{document.querySelector('.mode-controls select[aria-label="voice"]').focus();return true;})()`)
	pressKey(t, ctx, browser, "ArrowDown", 40)
	settle(`document.querySelector('.mode-controls select[aria-label="voice"]').value==='alba'`)
	check(`(()=>{const mode=document.querySelector('.composer select[aria-label="mode"]');mode.value='chat';mode.dispatchEvent(new Event('change'));mode.value='speech';mode.dispatchEvent(new Event('change'));return true;})()`)
	settle(`document.querySelector('.mode-controls select[aria-label="voice"]')?.value==='alba'`)
	check(`speechControl('max frames').value==='120' && speechControl('seed').value==='7' && speechControl('temperature').value==='0.7'`)
	check(`(()=>{document.querySelector('[data-draft-setting="temperature"]').value='-1';return true;})()`)
	check(`(()=>{document.querySelector('.composer .send-button').focus();return true;})()`)
	pressKey(t, ctx, browser, "Enter", 13)
	settle(`document.body.textContent.includes('speech fixture stopped before synthesis') && document.querySelector('.composer textarea').value==='Read this document aloud.'`)
	generator.mu.Lock()
	executions, speech, request, recipeUsed := generator.executions, generator.speech, generator.request, generator.recipeUsed
	generator.retired = true
	generator.mu.Unlock()
	if executions != 1 || recipeUsed != generator.secondRecipe || speech.Voice != "alba" || speech.MaxFrames != 120 || speech.Temperature != 0.7 || request.Seed == nil || *request.Seed != 7 {
		t.Fatalf("native workflow executions=%d recipe=%s speech=%+v request=%+v", executions, recipeUsed, speech, request)
	}
	for _, viewport := range webuilane.ScreenViewports {
		findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "speech-request-failed")
		if err != nil || len(findings) != 0 {
			t.Fatalf("failure %s: %v %v", viewport.Name, findings, err)
		}
	}
	check(`(()=>{const mode=document.querySelector('.composer select[aria-label="mode"]');mode.value='chat';mode.dispatchEvent(new Event('change'));mode.value='speech';mode.dispatchEvent(new Event('change'));return true;})()`)
	settle(`document.querySelector('.task-model-selector select[aria-label="generation model"]')?.selectedOptions[0]?.disabled && document.querySelector('.mode-controls').textContent.includes('selected model is unavailable')`)
	check(`document.querySelector('.composer textarea').value==='Read this document aloud.' && !document.querySelector('.mode-controls select[aria-label="voice"]') && !document.querySelector('[aria-label="Open speech options"]')`)
	for _, viewport := range webuilane.ScreenViewports {
		findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "speech-model-retired")
		if err != nil || len(findings) != 0 {
			t.Fatalf("retired %s: %v %v", viewport.Name, findings, err)
		}
	}
	check(`(()=>{document.querySelector('.task-model-selector select[aria-label="generation model"]').focus();return true;})()`)
	pressKey(t, ctx, browser, "ArrowDown", 40)
	settle(`!!document.querySelector('.mode-controls select[aria-label="voice"]') && !document.querySelector('.task-model-selector select[aria-label="generation model"]').selectedOptions[0].disabled`)
	check(`(async()=>{window.speechOriginalGet=overgo.api.get;window.speechCurrentCatalog=await overgo.api.get('/generation/capabilities');window.speechCatalogCalls=[];overgo.api.get=function(path,options){if(path!=='/generation/capabilities')return speechOriginalGet.call(this,path,options);return new Promise((resolve,reject)=>speechCatalogCalls.push({resolve,reject,signal:options.signal}));};window.speechMode=mode=>{const picker=document.querySelector('.composer select[aria-label="mode"]');picker.value=mode;picker.dispatchEvent(new Event('change'));};speechMode('chat');speechMode('speech');return true;})()`)
	settle(`speechCatalogCalls.length===1 && document.querySelector('.mode-controls').textContent.includes('Loading models')`)
	check(`(()=>{speechMode('chat');speechMode('speech');return speechCatalogCalls[0].signal.aborted;})()`)
	settle(`speechCatalogCalls.length===2`)
	check(`(()=>{speechCatalogCalls[0].resolve([]);return true;})()`)
	check(`!document.querySelector('.mode-controls select') && document.querySelector('.mode-controls').textContent.includes('Loading models')`)
	check(`(()=>{speechCatalogCalls[1].resolve(speechCurrentCatalog);return true;})()`)
	settle(`!!document.querySelector('.mode-controls select[aria-label="voice"]')`)
	check(`(()=>{speechMode('chat');speechMode('speech');return true;})()`)
	settle(`speechCatalogCalls.length===3`)
	check(`(()=>{speechCatalogCalls[2].reject(new Error('Speech catalog fixture unavailable'));return true;})()`)
	settle(`document.querySelector('.mode-controls').textContent.includes('Speech catalog fixture unavailable') && !!document.querySelector('.mode-controls button')`)
	check(`(()=>{document.querySelector('.mode-controls button').focus();return true;})()`)
	pressKey(t, ctx, browser, "Enter", 13)
	settle(`speechCatalogCalls.length===4`)
	check(`(()=>{speechCatalogCalls[3].resolve(speechCurrentCatalog);return true;})()`)
	settle(`!!document.querySelector('.mode-controls select[aria-label="voice"]')`)
	check(`document.querySelector('.composer textarea').value==='Read this document aloud.'`)
	check(`(()=>{speechMode('chat');speechMode('speech');return true;})()`)
	settle(`speechCatalogCalls.length===5`)
	check(`(()=>{window.speechOldComposer=document.querySelector('.composer');overgo.openConversation(null);return true;})()`)
	settle(`speechCatalogCalls[4].signal.aborted && speechCatalogCalls.length===6`)
	check(`(()=>{speechCatalogCalls[5].resolve(speechCurrentCatalog);return true;})()`)
	settle(`!!document.querySelector('.composer') && document.querySelector('.composer')!==speechOldComposer`)
	check(`(()=>{speechCatalogCalls[4].resolve([]);return true;})()`)
	check(`!document.querySelector('.mode-controls').textContent.includes('No model is available')`)
	check(`(()=>{overgo.api.get=speechOriginalGet;const input=document.querySelector('.composer textarea');input.value='Saved without a chat model';input.dispatchEvent(new Event('input'));return !document.querySelector('.composer').textContent.includes('Draft cannot be restored');})()`)
	if err := browser.Evaluate(ctx, `location.reload()`, nil); err != nil {
		t.Fatal(err)
	}
	settle(`document.querySelector('.composer textarea')?.value==='Saved without a chat model'`)
	check(`!document.querySelector('.composer').textContent.includes('Draft cannot be restored') && !document.querySelector('.composer').textContent.includes('server has not declared')`)
	t.Log("speech controls leg: keyboard voice/options/Close, required options and task-local validation, native selected recipe/voice request and failure recovery, retirement/reselection, catalog cancellation/stale refusal/retry/disposal; synthesis deliberately refused by fixture")
}
