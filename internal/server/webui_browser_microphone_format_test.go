package server

import (
	"encoding/base64"
	"net/http/httptest"
	"os"
	"overgo/internal/media"
	"overgo/internal/overgodb"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
	"strconv"
	"testing"
)

func TestWebUIBrowserMicrophoneFormat(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.ShortIntegration + ": microphone format runs through cmd/webui-lane")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	handler := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{pieces: []string{"Capture fixture"}}))
	front := httptest.NewServer(handler)
	defer front.Close()
	executable, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	browser, err := webuilane.Open(ctx, executable, front.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	check := func(expression string) { t.Helper(); assertBrowserPredicate(t, ctx, browser, expression) }
	settle := func(expression string) {
		t.Helper()
		if err := browser.Eventually(ctx, expression); err != nil {
			var state string
			_ = browser.Evaluate(ctx, `document.body.innerText`, &state)
			t.Fatalf("%s: %v; %s", expression, err, state)
		}
	}
	activate := func(label string) {
		t.Helper()
		check(`(()=>{const button=[...document.querySelectorAll('.capture-dialog button')].find(node=>node.textContent===` + strconv.Quote(label) + `&&!node.hidden);if(!button||button.disabled)return false;button.focus();return true;})()`)
		pressKey(t, ctx, browser, "Enter", 13)
	}
	settle(`!!document.querySelector('.composer textarea')`)
	check(`(()=>{
  window.micNativeContext=window.AudioContext;window.micContexts=[];window.micTracks=[];window.micRequests=[];window.micMode='normal';window.micFile=null;
  window.micDeclaration={format:{sample_rate:16000,channels:1,encoding:'pcm-f32le'},maximum_encoded_bytes:44+4096*2,maximum_samples:2048};
  window.AudioContext=function(options){micRequests.push(options?.sampleRate);if(micMode==='unsupported')throw new DOMException('Unsupported fixture rate','NotSupportedError');const context=new micNativeContext(micMode==='mismatch'?{sampleRate:24000}:options);micContexts.push(context);return context;};
  navigator.mediaDevices.getUserMedia=async()=>{const context=new micNativeContext({sampleRate:micDeclaration.format.sample_rate});await context.resume();const source=context.createOscillator(),destination=context.createMediaStreamDestination();source.connect(destination);source.start();const track=destination.stream.getAudioTracks()[0],stop=track.stop.bind(track);track.stop=()=>{stop();source.stop();context.close();};micTracks.push(track);if(micMode==='late')return new Promise(resolve=>window.micRelease=()=>resolve(destination.stream));return destination.stream;};
  navigator.mediaDevices.enumerateDevices=async()=>[];
  window.micOwned=overgo.composer(document.querySelector('#panel-chat'),{captureAccept:()=>['audio/wav'],captureAudio:()=>micDeclaration,intake:file=>{micFile=file;return Promise.resolve({id:'file:fixture'});}});
  window.micOpen=()=>micOwned.element.querySelector('.attach-button').click();micOwned.input.value='Keep this draft';return true;
 })()`)
	for _, rate := range []int{16000, 24000} {
		check(`(()=>{micDeclaration.format.sample_rate=` + strconv.Itoa(rate) + `;micDeclaration.maximum_samples=micDeclaration.format.sample_rate===16000?2048:4096;micDeclaration.maximum_encoded_bytes=44+(micDeclaration.format.sample_rate===16000?4096:2048)*2;micOpen();return true;})()`)
		check(`![...document.querySelectorAll('.capture-dialog button')].some(button=>button.getClientRects().length&&/photo|clip/i.test(button.textContent))`)
		activate("Record audio")
		activate("Record")
		settle(`document.querySelector('.capture-dialog').dataset.state==='ready'`)
		check(`micRequests.at(-1)===micDeclaration.format.sample_rate&&micContexts.every(c=>c.state==='closed')&&micTracks.every(t=>t.readyState==='ended')&&document.querySelector('.capture-dialog').textContent.includes('Recording limit reached.')`)
		for _, viewport := range append(append([]webuilane.Viewport(nil), webuilane.ScreenViewports...), webuilane.Viewport{Name: "phone-keyboard", Width: 390, Height: 480}) {
			findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "microphone-format-preview")
			if err != nil || len(findings) != 0 {
				t.Fatalf("%s: %v %v", viewport.Name, findings, err)
			}
		}
		activate("Attach recording")
		settle(`!!micFile`)
		var encoded string
		if err := browser.Evaluate(ctx, `(async()=>btoa(String.fromCharCode(...new Uint8Array(await micFile.arrayBuffer()))))()`, &encoded); err != nil {
			t.Fatal(err)
		}
		wave, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			t.Fatal(err)
		}
		decoded, _, err := media.DecodeAudio(ctx, wave, 2048)
		if err != nil {
			t.Fatal(err)
		}
		if decoded.Format.SampleRate != uint64(rate) || len(decoded.Samples) != 2048 {
			t.Fatalf("recording rate=%d samples=%d", decoded.Format.SampleRate, len(decoded.Samples))
		}
		check(`(()=>{micOwned.clearAttachments();micFile=null;return true;})()`)
	}
	for _, mode := range []string{"unsupported", "mismatch"} {
		check(`(()=>{micMode=` + strconv.Quote(mode) + `;micDeclaration.format.sample_rate=16000;micOpen();return true;})()`)
		activate("Record audio")
		activate("Record")
		settle(`document.querySelector('.capture-dialog').dataset.state==='error'`)
		check(`document.querySelector('.capture-dialog').textContent.includes('16000 Hz')&&micContexts.every(c=>c.state==='closed')&&micTracks.every(t=>t.readyState==='ended')`)
		activate("Close")
	}
	check(`(()=>{micMode='late';micOpen();return true;})()`)
	activate("Record audio")
	activate("Record")
	settle(`typeof micRelease==='function'`)
	check(`(()=>{micOwned.invalidateIntake();micRelease();return !document.querySelector('.capture-dialog');})()`)
	settle(`micContexts.every(c=>c.state==='closed')&&micTracks.every(t=>t.readyState==='ended')`)
	check(`(()=>{micOwned.dispose();micOwned.element.remove();window.AudioContext=micNativeContext;return overgo.errors.length===0;})()`)
	t.Log("microphone format leg: real AudioContext/worklet WAV at declared rates, sample limits, preview/attach, unsupported and differing actual rates, late permission and selection invalidation cleanup; synthetic streams, no physical microphone or phone")
}
