package audioparity

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"overgo/internal/server"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// Controlled microphone streams exercise the real browser recorder and native
// transcription. Isolated fixture activations do not promote production models.
func TestWebUIBrowserNativeASRMicrophone(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.ShortIntegration + ": native microphone acceptance runs through cmd/webui-lane")
	}
	f := newLiveASRFixture(t)
	// Leave room for browser setup/stop silence, bounded to twice the longest
	// source clip. This isolated fixture retains all native signal admission.
	policyPath := filepath.Join(filepath.Dir(f.path), "transcription_policy.json")
	policyBytes, err := os.ReadFile(policyPath)
	if err != nil {
		t.Fatal(err)
	}
	var policy server.TranscriptionPolicy
	if err := json.Unmarshal(policyBytes, &policy); err != nil {
		t.Fatal(err)
	}
	policy.Inspection.MaximumSamples *= 2
	policy.Inspection.MaximumEncodedBytes = 44 + 2*policy.Inspection.MaximumSamples
	policyBytes, err = json.Marshal(policy)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(policyPath, policyBytes, 0600); err != nil {
		t.Fatal(err)
	}
	_, front := f.proxy(t)
	browserURL := front.URL
	executable, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	for index, name := range []string{"granite", "nemotron"} {
		t.Run(name, func(t *testing.T) {
			want := strings.TrimSpace(f.transcribe(t, front, index))
			ctx, cancel := context.WithTimeoutCause(t.Context(), 90*time.Second, errors.New("real ASR browser journey did not settle"))
			defer cancel()
			browser, err := webuilane.Open(ctx, executable, browserURL)
			if err != nil {
				t.Fatal(err)
			}
			defer browser.Close()
			check := func(expression string) {
				t.Helper()
				var result bool
				if err := browser.Evaluate(ctx, expression, &result); err != nil || !result {
					t.Fatalf("browser check %s: %v", expression, err)
				}
			}
			settle := func(expression string) {
				t.Helper()
				if err := browser.Eventually(ctx, expression); err != nil {
					var state string
					_ = browser.Evaluate(ctx, `JSON.stringify({text:document.body.innerText,errors:overgo.errors})`, &state)
					t.Fatalf("unsettled %s: %v; %s", expression, err, state)
				}
			}
			settle(`!!document.querySelector('#api-key')`)
			check(`(()=>{document.querySelector('#settings-toggle').click();const key=document.querySelector('#api-key');key.value=` + strconv.Quote(audioHTTPFixtureKey) + `;key.dispatchEvent(new Event('change'));document.querySelector('#settings-dialog').close();return true;})()`)
			settle(`!!document.querySelector('.task-model-selector select') && !!document.querySelector('[aria-label="Transcription options"] input, .mode-controls input')`)
			check(`document.querySelector('.task-model-selector select').value===` + strconv.Quote(f.definitions[index].ID.String()))
			check(`(async()=>{
    const caps=await overgo.api.get('/generation/capabilities'),audio=caps.find(c=>c.recipe===document.querySelector('.task-model-selector select').value).controls.find(c=>c.name==='audio').audio;
    window.micTracks=[];window.micContexts=[];window.micEnded=false;
    navigator.mediaDevices.getUserMedia=async()=>{const context=new AudioContext({sampleRate:audio.format.sample_rate});micContexts.push(context);await context.resume();const bytes=Uint8Array.from(atob(` + strconv.Quote(base64.StdEncoding.EncodeToString(f.inputs[index])) + `),c=>c.charCodeAt(0));const decoded=await context.decodeAudioData(bytes.buffer),source=context.createBufferSource(),destination=context.createMediaStreamDestination();source.buffer=decoded;source.connect(destination);source.onended=()=>window.micEnded=true;window.micStart=()=>source.start();const track=destination.stream.getAudioTracks()[0],stop=track.stop.bind(track);track.stop=()=>{stop();context.close();};micTracks.push(track);return destination.stream;};
    navigator.mediaDevices.enumerateDevices=async()=>[];document.querySelector('.composer .attach-button').click();return true;
   })()`)
			activate := func(label string) {
				t.Helper()
				check(`(()=>{const button=[...document.querySelectorAll('.capture-dialog button')].find(b=>!b.hidden&&b.textContent===` + strconv.Quote(label) + `);if(!button||button.disabled)return false;button.focus();return true;})()`)
				for _, kind := range []string{"keyDown", "keyUp"} {
					event := map[string]any{"type": kind, "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13}
					if kind == "keyDown" {
						event["text"] = "\r"
					}
					if err := browser.Call(ctx, "Input.dispatchKeyEvent", event, nil); err != nil {
						t.Fatal(err)
					}
				}
			}
			activate("Record audio")
			activate("Record")
			settle(`document.querySelector('.capture-dialog').dataset.state==='recording'`)
			check(`(()=>{micStart();return true;})()`)
			settle(`micEnded`)
			activate("Stop recording")
			settle(`document.querySelector('.capture-dialog').dataset.state==='ready'`)
			check(`micTracks.every(t=>t.readyState==='ended')&&micContexts.every(c=>c.state==='closed')`)
			for _, viewport := range append(append([]webuilane.Viewport(nil), webuilane.ScreenViewports...), webuilane.Viewport{Name: "phone-keyboard", Width: 390, Height: 480}) {
				findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, name+"-microphone-preview")
				if err != nil || len(findings) != 0 {
					t.Fatalf("%s: %v %v", viewport.Name, findings, err)
				}
			}
			activate("Attach recording")
			settle(`document.querySelector('[aria-label="Transcription options"] input, .mode-controls input').value.startsWith('file:') && !document.querySelector('.composer .send-button').disabled`)
			settle(`!!document.querySelector('.attachment-preview audio') && document.querySelector('.attachment-preview audio').readyState>=2 && document.querySelector('.attachment-preview audio').duration>0`)
			if err := browser.Call(ctx, "Runtime.evaluate", map[string]any{"expression": `(()=>{const audio=document.querySelector('.attachment-preview audio');audio.muted=true;return audio.play();})()`, "userGesture": true, "awaitPromise": true}, nil); err != nil {
				t.Fatal(err)
			}
			settle(`document.querySelector('.attachment-preview audio').currentTime>0`)
			check(`(()=>{document.querySelector('.attachment-preview audio').pause();return true;})()`)
			for _, viewport := range append(append([]webuilane.Viewport(nil), webuilane.ScreenViewports...), webuilane.Viewport{Name: "phone-keyboard", Width: 390, Height: 480}) {
				findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, name+"-native-input")
				if err != nil || len(findings) != 0 {
					t.Fatalf("%s: %v %v", viewport.Name, findings, err)
				}
			}
			check(`(()=>{document.querySelector('.composer .send-button').focus();return true;})()`)
			for _, kind := range []string{"keyDown", "keyUp"} {
				event := map[string]any{"type": kind, "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13}
				if kind == "keyDown" {
					event["text"] = "\r"
				}
				if err := browser.Call(ctx, "Input.dispatchKeyEvent", event, nil); err != nil {
					t.Fatal(err)
				}
			}
			settle(`!!document.querySelector('.chat-log .msg.assistant .body') && document.querySelector('.chat-log .msg.assistant .body').textContent.trim()===` + strconv.Quote(want))
			check(`overgo.errors.length===0 && !document.body.textContent.includes('Cannot read properties')`)
			for _, viewport := range append(append([]webuilane.Viewport(nil), webuilane.ScreenViewports...), webuilane.Viewport{Name: "phone-keyboard", Width: 390, Height: 480}) {
				findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, name+"-native-transcription")
				if err != nil || len(findings) != 0 {
					t.Fatalf("%s: %v %v", viewport.Name, findings, err)
				}
			}
			t.Logf("native ASR microphone leg: real %s GUI: exact recipe %s, %d source-clip bytes recorded through synthetic microphone, keyboard submit, native transcript %q matches direct native HTTP; isolated model activations, not promotion or held-out quality", name, f.definitions[index].ID, len(f.inputs[index]), want)
		})
	}
}
