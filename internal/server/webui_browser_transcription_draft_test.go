package server

import (
	"context"
	"errors"
	"net/http/httptest"
	"os"
	"strconv"
	"testing"
	"time"

	"overgo/internal/overgodb"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserTranscriptionDraft: a recorded attachment transcribes
// through the production transcription route on explicit action, with
// progress and cancel, keeps the recording and the draft, offers insert or
// replace as an explicit choice, and a late or failed answer never touches
// the draft.
func TestWebUIBrowserTranscriptionDraft(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.ShortIntegration + ": transcription draft runs through cmd/webui-lane")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	handler := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{pieces: []string{"Transcription fixture"}}))
	defer handler.Close()
	server := httptest.NewServer(handler)
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeoutCause(t.Context(), 90*time.Second, errors.New("transcription draft journey did not settle"))
	defer cancel()
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
			_ = browser.Evaluate(ctx, `JSON.stringify({errors: window.overgo && window.overgo.errors, rows: [...document.querySelectorAll('.attachment-row')].map((row) => row.textContent.slice(0, 200)), draft: (document.querySelector('.composer textarea') || {}).value, dialog: (document.querySelector('.capture-dialog') || {}).textContent})`, &page)
			t.Fatalf("%s: %v; page: %s", expression, err, page)
		}
	}
	activate := func(text string) {
		t.Helper()
		check(`(() => {const button=[...document.querySelectorAll('.capture-dialog button')].find(node=>node.textContent===` + strconv.Quote(text) + `&&!node.hidden);if(!button||button.disabled)return false;button.focus();return document.activeElement===button;})()`)
		pressKey(t, ctx, browser, "Enter", 13)
	}
	settle(`!!document.querySelector('.composer textarea')`)
	// A synthetic microphone drives the real recorder; the served model declares transcription; the
	// route answers through the API client, which the journey observes and paces.
	check(`(() => {
  navigator.mediaDevices.getUserMedia=async()=>{
    const context=new AudioContext();await context.resume();const oscillator=context.createOscillator(),gain=context.createGain(),destination=context.createMediaStreamDestination();
    gain.gain.value=.25;oscillator.connect(gain).connect(destination);oscillator.start();const stream=destination.stream;
    const track=stream.getAudioTracks()[0],stop=track.stop.bind(track);track.stop=()=>{stop();context.close().catch(()=>{});};return stream;
  };
  navigator.mediaDevices.enumerateDevices=async()=>[{kind:'audioinput',deviceId:'fixture-mic',label:'Fixture microphone'}];
  const caps=structuredClone(overgo.capabilities());caps.modes.push({id:'transcription',label:'Transcribe',enabled:true});
  caps.media.accept=[...new Set([...(caps.media.accept||[]),'audio/wav'])];
  const get=overgo.api.get;
  overgo.api.get=async function(path,options){const result=await get.call(this,path,options);if(path==='/workspace/manifest')result.model=structuredClone(caps);return result;};
  window.transcriptCalls=[];window.transcriptMode='hold';window.transcriptSizes=[];
  const stream=overgo.api.stream;
  overgo.api.stream=async function(path,body,options){
    if(path!=='/v1/audio/transcriptions')return stream.call(this,path,body,options);
    const file=body.get('file');transcriptSizes.push(file.size);
    const bytes=new Uint8Array(await file.arrayBuffer());transcriptCalls.push({size:bytes.length,head:Array.from(bytes.slice(0,4)),type:file.type,signal:options.signal});
    if(transcriptMode==='fail'){const error=new Error('Transcription failed: the served model declined the audio');error.status=422;throw error;}
    return new Promise((resolve,reject)=>{
      options.signal.addEventListener('abort',()=>reject(new DOMException('aborted','AbortError')));
      window.releaseTranscript=(text)=>resolve(new Response(JSON.stringify({text}),{headers:{'Content-Type':'application/json'}}));
    });
  };
  window.captureOld=document.querySelector('.composer');overgo.openConversation(null);
  window.captureDialog=()=>document.querySelector('.capture-dialog');
  window.transcribeButton=()=>[...document.querySelectorAll('.attachment-row button')].find(button=>button.textContent==='Transcribe'&&!button.hidden);
  window.offerButton=(label)=>[...document.querySelectorAll('.attachment-row .transcript-offer button')].find(button=>button.textContent===label&&!button.hidden);
  window.rowStatus=()=>document.querySelector('.attachment-row [role=status], .attachment-row [role=alert]').textContent;
  return true;
})()`)
	settle(`!!document.querySelector('.composer .attach-button') && document.querySelector('.composer')!==captureOld`)
	check(`(() => {const input=document.querySelector('.composer textarea');input.value='Keep this draft';input.dispatchEvent(new Event('input'));document.querySelector('.composer .attach-button').click();return true;})()`)
	activate("Record audio")
	activate("Record")
	settle(`captureDialog().dataset.state==='recording' && captureDialog().querySelector('p').textContent.includes(' · ')`)
	activate("Stop recording")
	settle(`captureDialog().dataset.state==='ready'`)
	activate("Attach recording")
	settle(`document.querySelector('.attachment-row')?.dataset.state==='ready' && !!transcribeButton()`)

	// 1. Explicit action, progress and cancel: the draft and the recording stay.
	check(`(() => {transcribeButton().click();return true;})()`)
	settle(`transcriptCalls.length===1 && rowStatus().includes('Transcribing') && !!document.querySelector('.attachment-row button[aria-label^="Cancel"]:not([hidden])')`)
	check(`(() => {document.querySelector('.attachment-row button[aria-label^="Cancel"]').click();return true;})()`)
	settle(`transcriptCalls[0].signal.aborted && rowStatus().includes('stopped') && document.querySelector('.composer textarea').value==='Keep this draft' && document.querySelector('.attachment-row').dataset.state==='ready' && !!transcribeButton()`)
	// The bytes that left the page are the recording's: a WAV header on the attachment's own size.
	check(`transcriptCalls[0].type==='audio/wav' && transcriptCalls[0].head.join(',')==='82,73,70,70' && transcriptSizes[0]===Number(document.querySelector('.attachment-row').dataset.size)`)

	// 2. A transcript is an offer, never a send: insert keeps the draft, replace supersedes it.
	check(`(() => {transcribeButton().click();return true;})()`)
	settle(`transcriptCalls.length===2`)
	check(`(() => {releaseTranscript('the spoken words');return true;})()`)
	settle(`!!offerButton('Insert') && !!offerButton('Replace') && document.querySelector('.attachment-row .transcript-offer').textContent.includes('the spoken words') && document.querySelector('.composer textarea').value==='Keep this draft'`)
	check(`(() => {offerButton('Insert').click();return document.querySelector('.composer textarea').value==='Keep this draft\nthe spoken words' && !offerButton('Insert') && !document.querySelector('.send-button').disabled && document.querySelectorAll('#panel-chat .msg').length===0;})()`)
	check(`(() => {transcribeButton().click();return true;})()`)
	settle(`transcriptCalls.length===3`)
	check(`(() => {releaseTranscript('only these words');return true;})()`)
	settle(`!!offerButton('Replace')`)
	check(`(() => {offerButton('Replace').click();return document.querySelector('.composer textarea').value==='only these words' && !offerButton('Replace');})()`)

	// 3. A late answer for a removed recording changes nothing; a failure names the recovery.
	check(`(() => {transcribeButton().click();return true;})()`)
	settle(`transcriptCalls.length===4`)
	check(`(() => {const release=window.releaseTranscript;[...document.querySelectorAll('.attachment-row button')].find(button=>button.textContent==='Remove').click();release('a late answer');return true;})()`)
	check(`(() => {const input=document.querySelector('.composer textarea');input.value='Draft after removal';input.dispatchEvent(new Event('input'));return true;})()`)
	check(`document.querySelectorAll('.attachment-row').length===0 && document.querySelector('.composer textarea').value==='Draft after removal'`)
	check(`(() => {document.querySelector('.composer .attach-button').click();return true;})()`)
	activate("Record audio")
	activate("Record")
	settle(`captureDialog().dataset.state==='recording' && captureDialog().querySelector('p').textContent.includes(' · ')`)
	activate("Stop recording")
	settle(`captureDialog().dataset.state==='ready'`)
	activate("Attach recording")
	settle(`document.querySelector('.attachment-row')?.dataset.state==='ready' && !!transcribeButton()`)
	check(`(() => {window.transcriptMode='fail';transcribeButton().click();return true;})()`)
	settle(`rowStatus().includes('declined the audio') && !!transcribeButton() && document.querySelector('.composer textarea').value==='Draft after removal' && document.querySelector('.attachment-row').dataset.state==='ready'`)
	// Remount disposes the old composer: its transcription must stop, and a late answer stays detached.
	check(`(() => {window.transcriptMode='hold';window.priorTranscriptCount=transcriptCalls.length;transcribeButton().click();return true;})()`)
	settle(`transcriptCalls.length>priorTranscriptCount`)
	check(`(() => {window.disposedTranscription=transcriptCalls.at(-1).signal;window.disposedReply=releaseTranscript;window.oldTranscriptionComposer=document.querySelector('.composer');overgo.openConversation(null);return true;})()`)
	settle(`document.querySelector('.composer')!==oldTranscriptionComposer && !!document.querySelector('.composer textarea')`)
	check(`disposedTranscription.aborted`)
	check(`(() => {const input=document.querySelector('.composer textarea');input.value='Replacement draft';input.dispatchEvent(new Event('input'));disposedReply('Discard this late answer');return true;})()`)
	check(`document.querySelector('.composer textarea').value==='Replacement draft' && !document.querySelector('.transcript-offer:not([hidden])')`)
	t.Log("transcription draft leg: explicit transcribe with progress and cancel, verified recording bytes, insert or replace by choice, disposal and late answer preserve the draft")
}
