package server

import (
	"net/http/httptest"
	"os"
	"strconv"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserVideoCapture: the capture dialog records bounded camera
// clips only in a container the browser's recorder produces and the served
// input accepts, explains the mismatch otherwise while file and photo
// capture stay, never renames the container, bounds the clip's bytes,
// supports stop, preview, retake and discard, releases every track, and
// the attachment intake decides the format the way the server decodes it.
func TestWebUIBrowserVideoCapture(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.ShortIntegration + ": video capture runs through cmd/webui-lane")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	handler := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{pieces: []string{"Clip fixture"}}))
	handler.config.APIKey = testAPIKey
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
			var page string
			_ = browser.Evaluate(ctx, `JSON.stringify({errors: window.overgo && window.overgo.errors, dialog: (document.querySelector('.capture-dialog') || {}).textContent, state: (document.querySelector('.capture-dialog') || {dataset: {}}).dataset.state, rows: [...document.querySelectorAll('.attachment-row')].map((row) => row.textContent.slice(0, 160))})`, &page)
			t.Fatalf("%s: %v; page: %s", expression, err, page)
		}
	}
	activate := func(text string) {
		t.Helper()
		check(`(() => {const button=[...document.querySelectorAll('.capture-dialog button')].find(node=>node.textContent===` + strconv.Quote(text) + `&&!node.hidden);if(!button||button.disabled)return false;button.focus();return document.activeElement===button;})()`)
		pressKey(t, ctx, browser, "Enter", 13)
	}
	capture := func(state string) {
		t.Helper()
		for _, viewport := range []webuilane.Viewport{{Name: "desktop", Width: 1280, Height: 900}, {Name: "phone", Width: 390, Height: 844}, {Name: "phone-keyboard", Width: 390, Height: 380}, {Name: "small-phone", Width: 320, Height: 568}} {
			findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "camera-"+state)
			if err != nil {
				t.Fatal(err)
			}
			for _, finding := range findings {
				t.Errorf("camera %s: %+v", state, finding)
			}
			check(`(() => {const dialog=captureDialog(), bounds=dialog.getBoundingClientRect();return dialog.open && bounds.left>=0 && bounds.right<=innerWidth && bounds.top>=0 && bounds.bottom<=innerHeight && dialog.scrollWidth<=dialog.clientWidth && [...dialog.querySelectorAll('button,select')].filter(node=>!node.disabled && node.getClientRects().length && getComputedStyle(node).visibility!=='hidden').every(node=>{const r=node.getBoundingClientRect();return r.top>=0 && r.bottom<=innerHeight && r.left>=0 && r.right<=innerWidth;});})()`)
		}
	}
	settle(`!!document.querySelector('.composer textarea')`)
	check(`(() => {window.captureOld=document.querySelector('.composer');document.querySelector('#settings-toggle').click();const key=document.querySelector('#api-key');key.value=` + strconv.Quote(testAPIKey) + `;key.dispatchEvent(new Event('change'));document.querySelector('[aria-label="Close settings"]').click();return !document.querySelector('#settings-dialog').open;})()`)
	settle(`!!document.querySelector('.composer .attach-button') && document.querySelector('.composer')!==captureOld`)
	// A synthetic camera (a painted canvas) and microphone (an oscillator) drive the real recorder.
	check(`(() => {
  window.captureTracks=[];window.captureRequests=[];
  navigator.mediaDevices.getUserMedia=async constraints=>{
    captureRequests.push(structuredClone(constraints));
    const canvas=document.createElement('canvas');canvas.width=160;canvas.height=120;const stream=canvas.captureStream(10);
    const context=canvas.getContext('2d');let tick=0;window.capturePaint=setInterval(()=>{context.fillStyle=tick++%2?'rgb(60,110,220)':'rgb(220,110,60)';context.fillRect(0,0,canvas.width,canvas.height);},50);
    if(constraints.audio){const audio=new AudioContext();await audio.resume();const oscillator=audio.createOscillator(),destination=audio.createMediaStreamDestination();oscillator.connect(destination);oscillator.start();for(const track of destination.stream.getAudioTracks())stream.addTrack(track);}
    captureTracks.push(...stream.getTracks());
    return stream;
  };
  navigator.mediaDevices.enumerateDevices=async()=>[{kind:'videoinput',deviceId:'fixture-camera',label:'Fixture camera'}];
  window.recorderTypes=['video/mp4','video/webm'].filter(type=>window.MediaRecorder&&MediaRecorder.isTypeSupported(type));
  window.captureDialog=()=>document.querySelector('.capture-dialog');
  window.captureOpen=()=>document.querySelector('.composer .attach-button').click();
  window.clipButton=()=>[...captureDialog().querySelectorAll('button')].find(button=>button.textContent==='Record a clip');
  return recorderTypes.length>0;
})()`)

	// 1. The served input accepts no container this browser records: the choice explains both sides and stays disabled.
	check(`(() => {captureOpen();return true;})()`)
	settle(`!!clipButton()`)
	check(`(() => {const button=clipButton();const accepted=overgo.capabilities().media.accept.filter(type=>type.startsWith('video/'));const intersects=accepted.some(type=>recorderTypes.includes(type));return button.disabled===!intersects && (intersects || (button.title.includes(recorderTypes.join(', ')) && button.title.includes(accepted.length?accepted.join(', '):'no video')));})()`)
	check(`(() => {const files=[...captureDialog().querySelectorAll('button')].find(button=>button.textContent==='Choose files');const photo=[...captureDialog().querySelectorAll('button')].find(button=>button.textContent==='Take a photo');return !files.disabled && photo.disabled===!overgo.capabilities().media.accept.includes('image/png');})()`)
	activate("Close")
	t.Log("video capture negotiation leg: the choice states the browser's containers against the input's, file and photo capture stay")

	// 2. With a recordable container declared, a bounded clip records, previews, retakes and attaches in that container.
	check(`(() => {
  const caps=structuredClone(overgo.capabilities());caps.media.accept=[...new Set([...caps.media.accept,recorderTypes[0]])];caps.media.max_media_bytes=Math.max(caps.media.max_media_bytes,1<<20);
  const get=overgo.api.get;overgo.api.get=async function(path,options){const result=await get.call(this,path,options);if(path==='/workspace/manifest')result.model=structuredClone(caps);return result;};
  window.captureOld=document.querySelector('.composer');overgo.openConversation(null);return true;
})()`)
	settle(`!!document.querySelector('.composer .attach-button') && document.querySelector('.composer')!==captureOld`)
	check(`(() => {captureOpen();return true;})()`)
	settle(`!!clipButton() && !clipButton().disabled`)
	activate("Record a clip")
	settle(`captureDialog().dataset.state==='preview' && !!captureDialog().querySelector('video')`)
	check(`captureRequests.at(-1).video.facingMode.ideal==='user'`)
	for _, choice := range []string{"environment", "user", "device:fixture-camera"} {
		check(`(() => {window.priorCameraRequests=captureRequests.length;window.priorCameraTracks=captureTracks.filter(track=>track.readyState==='live');const device=captureDialog().querySelector('select');device.value=` + strconv.Quote(choice) + `;device.dispatchEvent(new Event('change'));return priorCameraTracks.length>0 && priorCameraTracks.every(track=>track.readyState==='ended');})()`)
		settle(`captureRequests.length===priorCameraRequests+1 && captureDialog().dataset.state==='preview'`)
		if choice == "device:fixture-camera" {
			check(`captureRequests.at(-1).video.deviceId.exact==='fixture-camera' && !captureRequests.at(-1).video.facingMode`)
		} else {
			check(`captureRequests.at(-1).video.facingMode.ideal===` + strconv.Quote(choice) + ` && !captureRequests.at(-1).video.deviceId`)
		}
		check(`captureDialog().querySelector('video').srcObject.getTracks().every(track=>track.readyState==='live')`)
	}
	capture("preview")
	activate("Record clip")
	settle(`captureDialog().dataset.state==='recording' && captureDialog().querySelector('p').textContent.includes(' · ')`)
	capture("recording")
	activate("Stop recording")
	settle(`captureDialog().dataset.state==='ready' && !!captureDialog().querySelector('video[controls]')`)
	check(`captureTracks.every(track=>track.readyState==='ended')`)
	settle(`captureDialog().querySelector('video').readyState>=HTMLMediaElement.HAVE_CURRENT_DATA`)
	if err := browser.Call(ctx, "Runtime.evaluate", map[string]any{"expression": `(async()=>{const video=captureDialog().querySelector('video');video.muted=true;await video.play();return true;})()`, "userGesture": true, "awaitPromise": true}, nil); err != nil {
		t.Fatal(err)
	}
	settle(`captureDialog().querySelector('video').currentTime>0 && !captureDialog().querySelector('video').paused`)
	check(`(() => {const video=captureDialog().querySelector('video');video.pause();return video.videoWidth>0 && video.videoHeight>0;})()`)
	capture("ready")
	activate("Retake")
	settle(`captureDialog().dataset.state==='preview'`)
	activate("Record clip")
	settle(`captureDialog().dataset.state==='recording' && captureDialog().querySelector('p').textContent.includes(' · ')`)
	activate("Stop recording")
	settle(`captureDialog().dataset.state==='ready'`)
	activate("Attach clip")
	settle(`document.querySelector('.attachment-row')?.dataset.state==='ready'`)
	check(`(() => {const row=document.querySelector('.attachment-row');const type=recorderTypes[0];return row.textContent.includes('clip.'+(type==='video/mp4'?'mp4':'webm')) && Number(row.dataset.size)<=overgo.capabilities().media.max_media_bytes && !!row.querySelector('video') && captureTracks.every(track=>track.readyState==='ended');})()`)
	t.Log("video capture recording leg: a bounded clip recorded, previewed, retook and attached in the negotiated container")

	// 3. The byte bound stops a clip that exceeds it, with a retake on offer.
	check(`(() => {overgo.capabilities().media.max_media_bytes=2048;captureOpen();return true;})()`)
	activate("Record a clip")
	settle(`captureDialog().dataset.state==='preview'`)
	activate("Record clip")
	settle(`captureDialog().dataset.state==='error' && captureDialog().textContent.includes('exceeded the recording limit')`)
	check(`captureTracks.every(track=>track.readyState==='ended')`)
	activate("Close")

	// 4. The intake decides the format the way the server decodes it: the recorded container is refused by name unless the server decodes it; a GIF is taken.
	check(`(async () => {
  const stored=overgo.capabilities().media.intake_accept||[];const type=recorderTypes[0];
  const clip=new File([new Uint8Array(64)],'clip.'+(type==='video/mp4'?'mp4':'webm'),{type});
  let outcome='';
  try{await overgo.api.upload('/artifacts/intake',clip);outcome='stored';}catch(err){outcome=String(err.message);}
  window.intakeOutcome=outcome;window.intakeDecodes=stored.includes(type);
  const gif=new File([Uint8Array.from(atob('R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'),c=>c.charCodeAt(0))],'frame.gif',{type:'image/gif'});
  const taken=await overgo.api.upload('/artifacts/intake',gif);window.intakeGIF=!!(taken&&taken.id);
  return true;
})()`)
	check(`intakeGIF && (intakeDecodes ? intakeOutcome==='stored' : intakeOutcome.includes('does not decode '+recorderTypes[0]))`)
	check(`(() => {clearInterval(window.capturePaint);return true;})()`)
	t.Log("video capture leg: negotiated container, bounded recording with stop, preview, retake and cleanup, and the intake's own decision on the format")
	t.Log("capture facing leg: front, rear and exact device constraints; previous tracks released before each switch; real recorder preview, recording and playback controls at desktop, phone, keyboard and small-phone sizes; synthetic devices, no physical-phone claim")
}
