package server

import (
	"context"
	"encoding/base64"
	"errors"
	"net/http/httptest"
	"os"
	"strconv"
	"testing"
	"time"

	"overgo/internal/media"
	"overgo/internal/overgodb"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

func TestWebUIBrowserMediaCapture(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.ShortIntegration + ": media capture runs through cmd/webui-lane")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	handler := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{pieces: []string{"Capture fixture"}}))
	handler.config.APIKey = testAPIKey
	server := httptest.NewServer(handler)
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeoutCause(t.Context(), 90*time.Second, errors.New("media capture journey did not settle"))
	defer cancel()
	browser, err := webuilane.Open(ctx, path, server.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	check := func(expression string) { t.Helper(); assertBrowserPredicate(t, ctx, browser, expression) }
	settle := func(expression string) {
		t.Helper()
		if err := browser.Eventually(ctx, "("+expression+") || document.querySelector('.capture-dialog')?.dataset.state==='error'"); err != nil {
			t.Fatalf("%s: %v", expression, err)
		}
		var result bool
		if err := browser.Evaluate(ctx, expression, &result); err != nil {
			t.Fatal(err)
		}
		if !result {
			var state string
			_ = browser.Evaluate(ctx, "document.querySelector('.capture-dialog')?.textContent", &state)
			t.Fatalf("%s: %s", expression, state)
		}
	}
	activate := func(text string) {
		t.Helper()
		check(`(() => {const button=[...document.querySelectorAll('.capture-dialog button')].find(node=>node.textContent===` + strconv.Quote(text) + `&&!node.hidden);if(!button||button.disabled)return false;button.focus();return true;})()`)
		pressKey(t, ctx, browser, "Enter", 13)
	}
	settle(`!!document.querySelector('.composer textarea')`)
	check(`(() => {window.captureOld=document.querySelector('.composer');document.querySelector('#settings-toggle').click();const key=document.querySelector('#api-key');key.value=` + strconv.Quote(testAPIKey) + `;key.dispatchEvent(new Event('change'));document.querySelector('#settings-dialog').close();return true;})()`)
	settle(`!!document.querySelector('.composer .attach-button') && document.querySelector('.composer')!==captureOld`)
	// Synthetic sources exercise the real AudioWorklet and canvas encoders. No
	// browser device permission or physical microphone/camera is used.
	check(`(() => {
  window.captureTracks=[];window.captureCalls=[];window.captureMode='normal';
  navigator.mediaDevices.getUserMedia=async constraints=>{
    captureCalls.push(constraints);
    if(captureMode==='denied')throw new DOMException('Fixture denial','NotAllowedError');
    let stream;
    if(constraints.audio){
      const context=new AudioContext();await context.resume();const oscillator=context.createOscillator(),gain=context.createGain(),destination=context.createMediaStreamDestination();
      gain.gain.value=.25;oscillator.connect(gain).connect(destination);oscillator.start();stream=destination.stream;
      const track=stream.getAudioTracks()[0],stop=track.stop.bind(track);track.stop=()=>{stop();context.close().catch(()=>{});};
    }else{
      const canvas=document.createElement('canvas');canvas.width=160;canvas.height=120;stream=canvas.captureStream();
      const context=canvas.getContext('2d');context.fillStyle='rgb(60,110,220)';context.fillRect(0,0,canvas.width,canvas.height);
    }
    captureTracks.push(...stream.getTracks());
    if(captureMode==='late')return new Promise(resolve=>window.releaseCapture=()=>resolve(stream));
    return stream;
  };
  navigator.mediaDevices.enumerateDevices=async()=>[{kind:'audioinput',deviceId:'fixture-mic',label:'Fixture microphone'},{kind:'videoinput',deviceId:'fixture-camera',label:'Fixture camera'}];
  window.captureGet=overgo.api.get;
  const caps=overgo.capabilities();caps.modes.push({id:'capture-fixture',label:'Capture fixture',enabled:true});
  const captureCaps=structuredClone(caps);
  overgo.api.get=async function(path,options){if(path==='/generation/capabilities')return [{task:'capture-fixture',recipe:'capture-fixture-recipe',name:'Synthetic media fixture',controls:[{name:'audio',label:'Audio input',type:'artifact',media:'audio'},{name:'image',label:'Image input',type:'artifact',media:'image'}]}];const result=await captureGet.call(this,path,options);if(path==='/workspace/manifest')result.model=structuredClone(captureCaps);return result;};
  window.captureOld=document.querySelector('.composer');overgo.openConversation(null);
  window.captureDialog=()=>document.querySelector('.capture-dialog');
  window.captureOpen=()=>document.querySelector('.composer .attach-button').click();
  window.captureSlot=kind=>[...document.querySelectorAll('.mode-controls label.control')].find(label=>label.textContent.includes(kind+' input')).querySelector('input');
  return true;
})()`)
	settle(`!!document.querySelector('.composer .attach-button') && document.querySelector('.composer')!==captureOld`)
	check(`(() => {captureOpen();return captureCalls.length===0&&[...captureDialog().querySelectorAll('button')].find(button=>button.textContent==='Record audio').disabled;})()`)
	activate("Close")
	check(`(() => {const mode=document.querySelector('.composer select[aria-label=mode]');mode.value='capture-fixture';mode.dispatchEvent(new Event('change'));return true;})()`)
	settle(`!!document.querySelector('.mode-controls label.control input')`)
	check(`(() => {const input=document.querySelector('.composer textarea');input.value='Preserve my draft';input.dispatchEvent(new Event('input'));window.captureByteLimit=overgo.capabilities().media.max_media_bytes;overgo.capabilities().media.max_media_bytes=44+2048*2;captureOpen();return captureCalls.length===0;})()`)
	activate("Record audio")
	check(`captureCalls.length===0`)
	activate("Record")
	settle(`captureDialog().dataset.state==='ready'`)
	check(`captureTracks.every(track=>track.readyState==='ended') && !!captureDialog().querySelector('audio[controls]') && captureCalls[0].video===false`)
	activate("Attach recording")
	settle(`!!captureSlot('Audio').value && document.querySelector('.attachment-row').dataset.state==='ready'`)
	var audioBase64 string
	if err := browser.Evaluate(ctx, `(async()=>{const blob=await overgo.api.blob('/artifacts/content?id='+encodeURIComponent(captureSlot('Audio').value));return btoa(String.fromCharCode(...new Uint8Array(await blob.arrayBuffer())));})()`, &audioBase64); err != nil {
		t.Fatal(err)
	}
	wave, err := base64.StdEncoding.DecodeString(audioBase64)
	if err != nil {
		t.Fatal(err)
	}
	decoded, _, err := media.DecodeAudio(ctx, wave, 2048)
	if err != nil {
		t.Fatalf("browser WAV: %v", err)
	}
	if len(decoded.Samples) != 2048 || len(wave) != 44+2048*2 {
		t.Fatalf("bounded WAV: %d samples, %d bytes", len(decoded.Samples), len(wave))
	}
	var positive, negative bool
	for _, sample := range decoded.Samples {
		positive = positive || sample > .1
		negative = negative || sample < -0.1
	}
	if !positive || !negative {
		t.Fatal("browser WAV did not preserve the synthetic waveform")
	}
	// Explicit Stop flushes the unfinished worklet chunk before preview.
	check(`(()=>{overgo.capabilities().media.max_media_bytes=captureByteLimit;captureOpen();return true;})()`)
	activate("Record audio")
	activate("Record")
	settle(`captureDialog().dataset.state==='recording' && captureDialog().querySelector('p').textContent.includes(' · ')`)
	activate("Stop recording")
	settle(`captureDialog().dataset.state==='ready'`)
	settle(`captureDialog().querySelector('audio').readyState>=HTMLMediaElement.HAVE_METADATA`)
	check(`captureDialog().querySelector('audio').duration>0 && captureTracks.every(track=>track.readyState==='ended')`)
	activate("Retake")
	activate("Record")
	settle(`captureDialog().dataset.state==='recording'`)
	activate("Close")
	check(`captureTracks.every(track=>track.readyState==='ended')`)
	check(`(()=>{const media=overgo.capabilities().media;media.max_image_dimension=80;media.max_image_pixels=80*60;return true;})()`)
	check(`(() => {captureOpen();return document.querySelector('.composer textarea').value==='Preserve my draft';})()`)
	activate("Take a photo")
	settle(`captureDialog().dataset.state==='preview'`)
	check(`captureCalls.at(-1).audio===false && captureDialog().querySelector('video').playsInline`)
	check(`(()=>{window.captureBeforeSwitch=captureTracks.slice();const device=captureDialog().querySelector('select');device.value='device:fixture-camera';device.dispatchEvent(new Event('change'));return true;})()`)
	settle(`captureDialog().dataset.state==='preview' && captureCalls.at(-1).video.deviceId?.exact==='fixture-camera'`)
	check(`captureBeforeSwitch.every(track=>track.readyState==='ended')`)
	for _, scheme := range webuilane.ColourSchemes {
		if err := browser.SetColorScheme(ctx, scheme); err != nil {
			t.Fatal(err)
		}
		for _, viewport := range []webuilane.Viewport{{Name: "phone", Width: 390, Height: 844}, {Name: "keyboard", Width: 390, Height: 320}, {Name: "small", Width: 320, Height: 568}} {
			if findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "media-capture-"+scheme); err != nil || len(findings) != 0 {
				t.Fatalf("capture %s layout: %v, %v", viewport.Name, findings, err)
			}
			check(`(()=>{const dialog=captureDialog().getBoundingClientRect(),button=[...captureDialog().querySelectorAll('button')].find(button=>button.textContent==='Take photo').getBoundingClientRect();return dialog.top>=0&&dialog.bottom<=innerHeight&&button.top>=0&&button.bottom<=innerHeight;})()`)
		}
	}
	activate("Take photo")
	settle(`captureDialog().dataset.state==='ready' && captureDialog().querySelector('img').naturalWidth>0`)
	if findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), webuilane.Viewport{Name: "small-keyboard", Width: 320, Height: 320}, "media-ready-light"); err != nil || len(findings) != 0 {
		t.Fatalf("ready capture layout: %v, %v", findings, err)
	}
	check(`(()=>{const buttons=[...captureDialog().querySelectorAll('.capture-actions button')].filter(button=>!button.hidden);return buttons.every(button=>{const rect=button.getBoundingClientRect();return rect.top>=0&&rect.bottom<=innerHeight&&rect.left>=0&&rect.right<=innerWidth;});})()`)
	check(`captureTracks.every(track=>track.readyState==='ended')`)
	activate("Retake")
	settle(`captureDialog().dataset.state==='preview'`)
	activate("Take photo")
	settle(`captureDialog().dataset.state==='ready'`)
	activate("Attach photo")
	settle(`!!captureSlot('Image').value && [...document.querySelectorAll('.attachment-row')].every(row=>row.dataset.state==='ready')`)
	check(`(async()=>{const blob=await overgo.api.blob('/artifacts/content?id='+encodeURIComponent(captureSlot('Image').value));const bitmap=await createImageBitmap(blob);const canvas=document.createElement('canvas');canvas.width=bitmap.width;canvas.height=bitmap.height;const context=canvas.getContext('2d');context.drawImage(bitmap,0,0);bitmap.close();const pixel=context.getImageData(0,0,1,1).data;return blob.type==='image/png'&&canvas.width===80&&canvas.height===60&&pixel[0]===60&&pixel[1]===110&&pixel[2]===220;})()`)
	// Encoded PNG bytes can exceed the file limit despite valid dimensions.
	// Derive a tighter byte limit from the captured fixture and verify the
	// next upload actually fits it, rather than repeatedly refusing a camera.
	check(`(async()=>{const blob=await overgo.api.blob('/artifacts/content?id='+encodeURIComponent(captureSlot('Image').value));window.captureImageID=captureSlot('Image').value;window.captureImageLimit=overgo.capabilities().media.max_image_bytes;window.captureTightLimit=Math.floor(blob.size*3/4);overgo.capabilities().media.max_image_bytes=captureTightLimit;captureOpen();return true;})()`)
	activate("Take a photo")
	settle(`captureDialog().dataset.state==='preview'`)
	activate("Take photo")
	settle(`captureDialog().dataset.state==='ready' && captureDialog().querySelector('img').naturalWidth>0`)
	check(`captureDialog().querySelector('img').naturalWidth<80`)
	activate("Attach photo")
	settle(`captureSlot('Image').value!==captureImageID`)
	check(`(async()=>{const blob=await overgo.api.blob('/artifacts/content?id='+encodeURIComponent(captureSlot('Image').value));overgo.capabilities().media.max_image_bytes=captureImageLimit;return blob.size<=captureTightLimit;})()`)
	// Denial stays actionable. A permission request which resolves after Close
	// must release its stream and never reopen or attach itself.
	check(`(()=>{captureMode='denied';captureOpen();return true;})()`)
	activate("Take a photo")
	settle(`captureDialog().dataset.state==='error' && captureDialog().textContent.includes('Access was denied')`)
	activate("Close")
	check(`(()=>{captureMode='late';captureOpen();return true;})()`)
	activate("Take a photo")
	settle(`typeof releaseCapture==='function'`)
	activate("Close")
	check(`(async()=>{releaseCapture();await Promise.resolve();await Promise.resolve();return !captureDialog()&&captureTracks.every(track=>track.readyState==='ended')&&document.querySelector('.composer textarea').value==='Preserve my draft';})()`)
	check(`(()=>{captureMode='normal';captureOpen();return true;})()`)
	activate("Take a photo")
	settle(`captureDialog().dataset.state==='preview'`)
	check(`(()=>{location.hash='#library';return true;})()`)
	settle(`!captureDialog() && captureTracks.every(track=>track.readyState==='ended')`)
	check(`(()=>{location.hash='#chat';return true;})()`)
	settle(`document.querySelector('.composer').closest('.panel').classList.contains('active')`)
	check(`(()=>{captureOpen();return document.querySelector('.composer textarea').value==='Preserve my draft';})()`)
	activate("Take a photo")
	settle(`captureDialog().dataset.state==='preview'`)
	check(`(()=>{const mode=document.querySelector('.composer select[aria-label=mode]');mode.value='chat';mode.dispatchEvent(new Event('change'));return !captureDialog()&&captureTracks.every(track=>track.readyState==='ended');})()`)
	// The shared composer also owns read-only/disposal cleanup for consumers
	// which keep their mounted panel while changing selection or input state.
	check(`(()=>{window.captureOwned=overgo.composer(document.querySelector('#panel-chat'),{captureAccept:()=>['image/png']});captureOwned.input.value='Owned draft';captureOwned.element.querySelector('.attach-button').click();return true;})()`)
	activate("Take a photo")
	settle(`captureDialog().dataset.state==='preview'`)
	check(`(()=>{captureOwned.setReadOnly(true);captureOwned.element.querySelector('.attach-button').click();return !captureDialog()&&captureTracks.every(track=>track.readyState==='ended')&&captureOwned.input.readOnly&&captureOwned.input.value==='Owned draft';})()`)
	check(`(()=>{captureOwned.setReadOnly(false);captureOwned.element.querySelector('.attach-button').click();return true;})()`)
	activate("Take a photo")
	settle(`captureDialog().dataset.state==='preview'`)
	pressKey(t, ctx, browser, "Escape", 27)
	check(`!captureDialog()&&captureTracks.every(track=>track.readyState==='ended')`)
	check(`(()=>{captureOwned.element.querySelector('.attach-button').click();return true;})()`)
	activate("Take a photo")
	settle(`captureDialog().dataset.state==='preview'`)
	check(`(()=>{captureOwned.dispose();captureOwned.element.querySelector('.attach-button').click();captureOwned.element.remove();return !captureDialog()&&captureTracks.every(track=>track.readyState==='ended');})()`)
	check(`overgo.errors.length===0`)
	t.Log("media capture leg: synthetic AudioWorklet WAV with independent decoder, byte-limit auto-stop and explicit Stop; camera switching, preview/retake and resized PNG pixels; authenticated artifact intake; declared-slot admission; permission denial, late permission cleanup, navigation, mode/read-only/disposal and Escape cleanup; draft preservation and narrow/keyboard viewport controls passed. No physical microphone/camera or model-understanding claim.")
}
