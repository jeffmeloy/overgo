package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
	"testing"
	"time"
)

// Expose only workflows: this fixture has no loaded text model or model properties.
type coldStartWorkflow struct {
	GenerationRefused
	WorkflowWorkspaceAPI
}

func coldStartSpeechFixture(t *testing.T) (*IdleShell, *speechSurfaceGenerator) {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	generator := &speechSurfaceGenerator{retired: true, generationWorkspaceGenerator: &generationWorkspaceGenerator{fakeGenerator: &fakeGenerator{}, repository: store, run: testutil.ArtifactID(t, artifact.KindRun, "cold-speech-run"), capability: WorkflowCapability{
		Task: recipe.TaskSpeech, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "cold-speech-recipe"), Name: "Local speech (UI fixture)", Stages: []recipe.Stage{{Node: recipe.Node{ID: "generate", Module: "test.generate"}}}, Controls: []WorkflowControl{{Name: "text", Type: WorkflowControlText, Required: true}, {Name: "voice", Type: WorkflowControlText, Required: true, Choices: []string{"alba", "marius"}}},
	}}}
	handler, err := New(Config{Repository: store, RuntimePolicy: testRuntimePolicy()}, &coldStartWorkflow{GenerationRefused: GenerationRefused{Reason: "No chat model is loaded"}, WorkflowWorkspaceAPI: generator})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = handler.Close() })
	return &IdleShell{Repository: store, Workbench: handler}, generator
}

func TestAudioStartupCapabilities(t *testing.T) {
	shell, _ := coldStartSpeechFixture(t)
	response := httptest.NewRecorder()
	shell.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/workspace/manifest", nil))
	var manifest workspaceManifestResponse
	if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &manifest) != nil {
		t.Fatalf("manifest: %d %s", response.Code, response.Body)
	}
	if manifest.Model == nil || manifest.Model.Model != "" || manifest.Model.Recipe != "" || manifest.Model.Name != "" {
		t.Fatalf("no text model must be invented: %+v", manifest.Model)
	}
	for _, mode := range manifest.Model.Modes {
		if mode.Enabled != (mode.ID == "speech") {
			t.Fatalf("unexpected enabled mode: %+v", mode)
		}
	}
	found := false
	for _, tab := range manifest.Tabs {
		if tab.ID == "chat" {
			found = tab.Enabled
		}
		if tab.ID == "model" && tab.Enabled {
			t.Fatal("model properties enabled without model")
		}
	}
	if !found {
		t.Fatal("main workbench is unreachable with admitted speech")
	}
	response = httptest.NewRecorder()
	shell.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/generation/capabilities", nil))
	var available []WorkflowCapability
	if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &available) != nil || len(available) != 1 || available[0].Task != recipe.TaskSpeech {
		t.Fatalf("speech catalog: %d %s", response.Code, response.Body)
	}
}

func TestWebUIBrowserAudioStartup(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.ShortIntegration + ": cold audio browser probe")
	}
	shell, generator := coldStartSpeechFixture(t)
	server := httptest.NewServer(shell)
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeoutCause(t.Context(), 45*time.Second, errors.New("cold speech browser did not settle"))
	defer cancel()
	browser, err := webuilane.Open(ctx, path, server.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	settle := func(predicate string) {
		t.Helper()
		if err := browser.Eventually(ctx, predicate); err != nil {
			var state string
			_ = browser.Evaluate(t.Context(), `JSON.stringify({text:document.body.innerText,errors:overgo.errors})`, &state)
			t.Fatalf("%v; %s", err, state)
		}
	}
	check := func(predicate string) { t.Helper(); assertBrowserPredicate(t, ctx, browser, predicate) }
	settle(`!!document.querySelector('.mode-controls select[aria-label="voice"]')`)
	check(`!document.querySelector('.composer select[aria-label="mode"]') && !overgo.capabilities().recipe && !document.querySelector('#cold-start')`)
	check(`(()=>{document.querySelector('.mode-controls select[aria-label="voice"]').focus();return true;})()`)
	pressKey(t, ctx, browser, "ArrowDown", 40)
	check(`(()=>{document.querySelector('.composer textarea').focus();return true;})()`)
	if err := browser.Call(ctx, "Input.insertText", map[string]any{"text": "Speech without loading chat."}, nil); err != nil {
		t.Fatal(err)
	}
	for _, viewport := range append(webuilane.ScreenViewports, webuilane.Viewport{Name: "small-phone", Width: 320, Height: 568}, webuilane.Viewport{Name: "phone-landscape", Width: 844, Height: 390}) {
		findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "cold-speech-ready")
		if err != nil || len(findings) != 0 {
			t.Fatalf("%s: %v %v", viewport.Name, findings, err)
		}
	}
	check(`(()=>{
   window.taskHeaderPicker=document.querySelector('.task-model-selector select');
   if(!taskHeaderPicker || !document.querySelector('#model-pill').hidden || document.querySelectorAll('[aria-label="generation model"]').length!==1)return false;
   const choose=[...document.querySelectorAll('.front-empty button')].find(button=>button.textContent==='Choose a model');
   if(!choose)return false;
   choose.click();
   if(document.activeElement!==taskHeaderPicker || document.querySelector('dialog[open]'))return false;
   window.taskHeaderAPI=overgo.api.get;window.taskHeaderHealth=0;
   overgo.api.get=async function(path,options){const answer=await taskHeaderAPI.call(this,path,options);if(path==='/health'){taskHeaderHealth++;return {...answer,model:'unrelated chat health'};}return answer;};
   return true;
 })()`)
	settle(`taskHeaderHealth>0`)
	check(`(()=>{overgo.api.get=taskHeaderAPI;return document.querySelector('#model-pill').hidden && document.querySelector('.task-model-selector select')===taskHeaderPicker && document.querySelector('#model-details').textContent==='Local speech (UI fixture)' && document.querySelector('.composer textarea').value==='Speech without loading chat.';})()`)
	check(`(()=>{location.hash='#library';return true;})()`)
	settle(`!document.querySelector('#model-pill').hidden && document.querySelector('.task-model-selector').hidden`)
	check(`(()=>{location.hash='#chat';return true;})()`)
	settle(`document.querySelector('#model-pill').hidden && document.querySelector('.task-model-selector select')===taskHeaderPicker && document.querySelector('.composer textarea').value==='Speech without loading chat.'`)
	check(`(()=>{
   window.taskHeaderPost=overgo.api.post;
   overgo.api.post=function(path,body,options){if(path!=='/generation/run')return taskHeaderPost.call(this,path,body,options);return new Promise(resolve=>{window.taskHeaderSubmission={body,release:()=>resolve(taskHeaderPost.call(overgo.api,path,body,options))};});};
   document.querySelector('.composer .send-button').focus();return true;
 })()`)
	pressKey(t, ctx, browser, "Enter", 13)
	settle(`!!window.taskHeaderSubmission && !document.querySelector('.composer .stop-button').hidden`)
	check(`(()=>{document.querySelector('.mode-controls select[aria-label="voice"]').focus();return true;})()`)
	pressKey(t, ctx, browser, "ArrowDown", 40)
	check(`(()=>{if(taskHeaderSubmission.body.input.voice!=='alba' || document.querySelector('.mode-controls select[aria-label="voice"]').value!=='marius')return false;overgo.api.post=taskHeaderPost;taskHeaderSubmission.release();return true;})()`)
	settle(`document.body.textContent.includes('speech fixture stopped before synthesis') && document.querySelector('.composer textarea').value==='Speech without loading chat.'`)
	generator.mu.Lock()
	defer generator.mu.Unlock()
	if generator.executions != 1 || generator.recipeUsed != generator.capability.Recipe || generator.speech.Voice != "alba" {
		t.Fatalf("wrong cold-start request: executions=%d recipe=%s voice=%q", generator.executions, generator.recipeUsed, generator.speech.Voice)
	}
	t.Log("audio startup leg: native idle manifest and single-task first mount select and submit admitted speech without a chat model; fixture refuses synthesis")
}
