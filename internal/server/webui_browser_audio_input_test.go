package server

import (
	"encoding/base64"
	"net/http"
	"net/http/httptest"
	"os"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
)

func TestWebUIBrowserAudioInput(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.ShortIntegration + ": isolated native transcription startup probe")
	}
	for _, earlyKey := range []bool{true, false} {
		t.Run("early-key="+strconv.FormatBool(earlyKey), func(t *testing.T) {
			fixture := newTranscriptionHTTPFixture(t, nil)
			handler, err := New(Config{Repository: fixture.store, RuntimePolicy: testRuntimePolicy(), APIKey: testAPIKey}, &coldStartWorkflow{GenerationRefused: GenerationRefused{Reason: "no chat model is loaded"}, WorkflowWorkspaceAPI: fixture.workspace})
			if err != nil {
				t.Fatal(err)
			}
			defer handler.Close()
			initial := make(chan struct{})
			held := make(chan struct{})
			var first atomic.Bool
			release := sync.OnceFunc(func() { close(held) })
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/workspace/manifest" && first.CompareAndSwap(false, true) {
					close(initial)
					select {
					case <-held:
					case <-r.Context().Done():
						return
					}
				}
				handler.ServeHTTP(w, r)
			}))
			defer server.Close()
			defer release()
			path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
			if err != nil {
				t.Fatal(err)
			}
			ctx := t.Context()
			browser, err := webuilane.Open(ctx, path, "about:blank")
			if err != nil {
				t.Fatal(err)
			}
			defer browser.Close()
			if err := browser.Call(ctx, "Page.navigate", map[string]any{"url": server.URL}, nil); err != nil {
				t.Fatal(err)
			}
			check := func(p string) { t.Helper(); assertBrowserPredicate(t, ctx, browser, p) }
			settle := func(p string) {
				t.Helper()
				if err := browser.Eventually(ctx, p); err != nil {
					var state string
					_ = browser.Evaluate(t.Context(), `JSON.stringify({text:document.body.innerText,errors:overgo.errors})`, &state)
					t.Fatalf("%v; %s", err, state)
				}
			}
			select {
			case <-initial:
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			}
			enterKey := func() {
				check(`(()=>{document.querySelector('#settings-toggle').click();const key=document.querySelector('#api-key');key.value=` + strconv.Quote(testAPIKey) + `;key.dispatchEvent(new Event('change'));document.querySelector('#settings-dialog').close();return overgo.getKey()===key.value;})()`)
			}
			if earlyKey {
				enterKey()
			}
			release()
			if !earlyKey {
				settle(`document.querySelector('#panels')?.textContent.includes('API key required')`)
				check(`!document.querySelector('#panels').textContent.includes('Start the server') && !document.querySelector('#panels').textContent.includes('no server at')`)
				enterKey()
			}
			settle(`!!document.querySelector('.task-model-selector select') && !!document.querySelector('[aria-label="Transcription options"] input')`)
			check(`!document.querySelector('.composer select[aria-label="mode"]') && overgo.capabilities().modes.filter(mode=>mode.enabled).every(mode=>mode.id==='transcription')`)
			check(`(()=>{const welcome=document.querySelector('.front-empty');const ask=[...welcome.querySelectorAll('button')].find(button=>button.textContent==='Ask a question');return welcome.querySelector('h2').textContent==='Transcribe audio' && ask.hidden && !ask.getClientRects().length;})()`)
			check(`(()=>{window.asrFetch=window.fetch;window.fetch=async function(url,options){if(String(url).includes('/artifacts/intake')&&!window.asrHeld){window.asrHeld=true;window.asrSignal=options.signal;const response=await window.asrFetch(url,options);window.asrOldID=(await response.clone().json()).id;return new Promise(resolve=>{window.releaseASR=()=>resolve(response);});}return window.asrFetch(url,options);};return true;})()`)
			check(`(()=>{const bytes=Uint8Array.from(atob(` + strconv.Quote(base64.StdEncoding.EncodeToString(fixture.wave)) + `),c=>c.charCodeAt(0)),file=new File([bytes.map((value,index)=>index===bytes.length-1?value^1:value)],'older.wav',{type:'audio/wav'}),transfer=new DataTransfer();transfer.items.add(file);const picker=document.querySelector('.composer input[type=file]');picker.files=transfer.files;picker.dispatchEvent(new Event('change'));return true;})()`)
			settle(`!!window.releaseASR`)
			check(`(()=>{const bytes=Uint8Array.from(atob(` + strconv.Quote(base64.StdEncoding.EncodeToString(fixture.wave)) + `),c=>c.charCodeAt(0)),file=new File([bytes],'recording.wav',{type:'audio/wav'}),transfer=new DataTransfer();transfer.items.add(file);const picker=document.querySelector('.composer input[type=file]');picker.files=transfer.files;picker.dispatchEvent(new Event('change'));return true;})()`)
			settle(`document.querySelector('[aria-label="Transcription options"] input').value.startsWith('file:') && !document.querySelector('.composer .send-button').disabled`)
			check(`(()=>{window.asrSelected=document.querySelector('[aria-label="Transcription options"] input').value;window.releaseASR();return window.asrSignal.aborted && window.asrSelected!==window.asrOldID && document.querySelectorAll('.attachment-row').length===1 && document.querySelector('.attachment-name').textContent==='recording.wav';})()`)
			settle(`document.querySelector('[aria-label="Transcription options"] input').value===window.asrSelected && !document.querySelector('.composer .send-button').disabled`)
			settle(`!!document.querySelector('.attachment-preview audio') && document.querySelector('.attachment-preview audio').readyState>=1`)
			check(`document.querySelector('.transcription-controls').hidden && [...document.querySelectorAll('.attachment-actions button')].every(button=>button.hidden || !button.textContent.startsWith('Transcribe')) && document.querySelector('.composer .send-button').textContent==='Transcribe' && !document.querySelector('.mode-controls input')`)
			for _, viewport := range webuilane.ScreenViewports {
				findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "cold-transcription-ready")
				if err != nil || len(findings) != 0 {
					t.Fatalf("%s: %v %v", viewport.Name, findings, err)
				}
			}
			check(`(()=>{window.asrPost=overgo.api.post;overgo.api.post=function(path,body,options){if(path==='/generation/run')window.asrSubmitted=body;return window.asrPost.call(this,path,body,options);};document.querySelector('.composer .send-button').focus();return true;})()`)
			pressKey(t, ctx, browser, "Enter", 13)
			settle(`!!document.querySelector('.chat-log .msg.assistant .body') && document.querySelector('.chat-log .msg.assistant .body').textContent.trim()==='x'`)
			for _, viewport := range webuilane.ScreenViewports {
				findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "cold-transcription-result")
				if err != nil || len(findings) != 0 {
					t.Fatalf("%s: %v %v", viewport.Name, findings, err)
				}
			}
			check(`asrSubmitted.task==='transcription' && asrSubmitted.input.audio===asrSelected && document.querySelector('.task-model-selector select').value===asrSubmitted.recipe`)
			t.Log("audio input leg: one declared ASR task, WAV attachment intake and native recipe execution produce fixture transcript x without chat model properties; fixture encoder only, not production ASR quality")
		})
	}
}
