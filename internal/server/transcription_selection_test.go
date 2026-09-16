package server

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strconv"
	"sync/atomic"
	"testing"

	"overgo/internal/operation"
	"overgo/internal/runrecord"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

func TestTranscriptionSelectedRecipe(t *testing.T) {
	fixture := newTranscriptionHTTPFixture(t, nil)
	request := httptest.NewRequest(http.MethodGet, "/generation/capabilities", nil)
	request.Header.Set("Authorization", "Bearer "+testAPIKey)
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	var choices []WorkflowCapability
	if err := json.Unmarshal(response.Body.Bytes(), &choices); err != nil || response.Code != http.StatusOK || len(choices) != 1 {
		t.Fatalf("capabilities HTTP=%d body=%s decode=%v", response.Code, response.Body, err)
	}
	if choices[0].Recipe != fixture.workspace.policy.Recipe || choices[0].Refusal != "" {
		t.Fatalf("wrong admitted recipe: %+v", choices)
	}
	for _, test := range []struct {
		name, model string
		status      int
	}{
		{"selected recipe", choices[0].Recipe.String(), http.StatusOK},
		{"missing selection", "", http.StatusBadRequest},
		{"unrelated chat model", fixture.handler.modelArtifact.String(), http.StatusBadRequest},
	} {
		t.Run(test.name, func(t *testing.T) {
			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, fixture.request(t, fixture.wave, map[string]string{"model": test.model}))
			if response.Code != test.status {
				t.Fatalf("HTTP=%d body=%s", response.Code, response.Body)
			}
		})
	}
	statuses := fixture.handler.operations.List()
	if len(statuses) != 1 || statuses[0].State != operation.StateCompleted {
		t.Fatalf("unexpected operations: %+v", statuses)
	}
	run, err := runrecord.RequireExactRun(t.Context(), fixture.store, *statuses[0].Run)
	if err != nil || run.Recipe != choices[0].Recipe {
		t.Fatalf("selected run=%+v err=%v", run, err)
	}
}

func TestWebUIBrowserTranscriptionSelection(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.ShortIntegration + ": transcription selection runs through cmd/webui-lane")
	}
	fixture := newTranscriptionHTTPFixture(t, nil)
	// The chat attachment declaration allows WAV; transcription itself uses the
	// independent, admitted native ASR fixture and never this projector.
	fixture.handler.config.AudioProjector = &fakeAudioProjector{}
	fixture.handler.config.ModelID = "unrelated-chat-model"
	fixture.handler.config.APIKey = ""
	var failCatalog atomic.Bool
	failCatalog.Store(true)
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/generation/capabilities" && failCatalog.CompareAndSwap(true, false) {
			writeError(w, http.StatusServiceUnavailable, "fixture_unavailable", "Transcription models could not be loaded.")
			return
		}
		if r.URL.Path == "/v1/audio/transcriptions" {
			select {
			case <-release:
			case <-r.Context().Done():
				return
			}
		}
		fixture.handler.ServeHTTP(w, r)
	}))
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
			var state string
			_ = browser.Evaluate(ctx, `JSON.stringify({page:document.body.innerText,errors:overgo.errors})`, &state)
			t.Fatalf("%s: %v; %s", expression, err, state)
		}
	}
	settle(`!!document.querySelector('.composer textarea')`)
	check(`(() => {const input=document.querySelector('.composer textarea');input.value='Keep my draft';input.dispatchEvent(new Event('input'));const bytes=Uint8Array.from(atob(` + strconv.Quote(base64.StdEncoding.EncodeToString(fixture.wave)) + `),c=>c.charCodeAt(0));const transfer=new DataTransfer();transfer.items.add(new File([bytes],'spoken-note.wav',{type:'audio/wav'}));document.querySelector('.composer').dispatchEvent(new DragEvent('drop',{dataTransfer:transfer,bubbles:true}));return true;})()`)
	capture := func(state string) {
		t.Helper()
		for _, viewport := range slices.Concat(webuilane.ScreenViewports, []webuilane.Viewport{{Name: "phone-keyboard", Width: 390, Height: 380}}) {
			if viewport.Name == "phone-keyboard" && state == "transcription-offer" {
				if _, err := webuilane.CaptureState(ctx, browser, "", viewport, state); err != nil {
					t.Fatal(err)
				}
				check(`(() => {document.querySelector('.transcript-offer button').focus();return true;})()`)
			}
			if findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, state); err != nil || len(findings) != 0 {
				t.Fatalf("%s %s: findings=%v err=%v", viewport.Name, state, findings, err)
			}
			if state == "transcription-offer" {
				check(`(() => {const buttons=[...document.querySelectorAll('.transcript-offer button'),document.querySelector('.send-button')];return buttons.every(button=>{const r=button.getBoundingClientRect();return r.width>0&&r.height>0&&r.top>=0&&r.bottom<=innerHeight&&document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)===button;});})()`)
			}
		}
	}
	settle(`!!document.querySelector('.transcription-controls button:not([hidden])') && document.querySelector('.transcription-controls [role=status]').textContent.includes('could not be loaded')`)
	capture("transcription-model-error")
	check(`(() => {document.querySelector('.transcription-controls button').click();return document.querySelector('.composer textarea').value==='Keep my draft';})()`)
	settle(`document.querySelector('.attachment-row')?.dataset.state==='ready' && !!document.querySelector('.attachment-actions button:not([hidden]):not([disabled])[aria-label^="Transcribe "]')`)
	check(`document.querySelector('[aria-label="Transcription model"]').value===` + strconv.Quote(fixture.workspace.policy.Recipe.String()))
	capture("transcription-ready")
	for _, action := range []string{"Discard", "Insert", "Replace"} {
		check(`(() => {const button=document.querySelector('.attachment-actions button[aria-label^="Transcribe "]');button.focus();return document.activeElement===button;})()`)
		pressKey(t, ctx, browser, "Enter", 13)
		if action == "Discard" {
			settle(`document.querySelector('.attachment-row [role=status]').textContent==='Transcribing…'`)
			capture("transcription-pending")
			close(release)
		}
		settle(`!!document.querySelector('.transcript-offer:not([hidden])') && document.querySelector('.transcript-text').textContent==='x'`)
		if action == "Discard" {
			capture("transcription-offer")
		}
		check(`(() => {const button=[...document.querySelectorAll('.transcript-offer button')].find(button=>button.textContent===` + strconv.Quote(action) + `);button.focus();return document.activeElement===button;})()`)
		pressKey(t, ctx, browser, "Enter", 13)
		want := "Keep my draft"
		if action == "Insert" {
			want += "\nx"
		}
		if action == "Replace" {
			want = "x"
		}
		check(`document.querySelector('.composer textarea').value===` + strconv.Quote(want) + ` && !document.querySelector('.transcript-offer:not([hidden])')`)
	}
	check(`document.querySelectorAll('.attachment-row').length===1 && overgo.errors.length===0`)
	statuses := fixture.handler.operations.List()
	if len(statuses) != 3 {
		t.Fatalf("native request count=%d, want 3", len(statuses))
	}
	for _, status := range statuses {
		if status.State != operation.StateCompleted || status.Run == nil {
			t.Fatalf("native operation=%+v", status)
		}
		run, err := runrecord.RequireExactRun(t.Context(), fixture.store, *status.Run)
		if err != nil || run.Recipe != fixture.workspace.policy.Recipe {
			t.Fatalf("native recipe=%+v err=%v", run, err)
		}
	}
	t.Log("transcription selection leg: actual multipart handler and native ASR fixture; exact recipe and output lineage, retained recording, explicit discard/insert/replace and desktop/mobile graphical states")
}
