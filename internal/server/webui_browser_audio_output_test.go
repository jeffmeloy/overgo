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

	"overgo/internal/testevidence"
	"overgo/internal/webuilane"
)

func TestWebUIBrowserAudioOutput(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testevidence.ShortIntegrationSkip + ": authenticated media runs through cmd/webui-lane")
	}
	fixture := newTranscriptionHTTPFixture(t, nil)
	fixture.handler.config.APIKey = testAPIKey
	server := httptest.NewServer(fixture.handler)
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeoutCause(t.Context(), 60*time.Second, errors.New("audio output journey did not settle"))
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
			t.Fatalf("%s: %v", expression, err)
		}
	}
	settle(`!!document.querySelector('#settings-toggle') && !!document.querySelector('.composer textarea')`)
	check(`(() => {const thread=overgo.thread;overgo.thread=(...args)=>(window.authMediaThread=thread(...args));window.authMediaOld=document.querySelector('.composer');document.querySelector('#settings-toggle').click();const key=document.querySelector('#api-key');key.value=` + strconv.Quote(testAPIKey) + `;key.dispatchEvent(new Event('change'));document.querySelector('#settings-dialog').close();return true;})()`)
	settle(`!!window.authMediaThread && document.querySelector('.composer')!==authMediaOld`)
	check(`(async () => {const bytes=Uint8Array.from(atob(` + strconv.Quote(base64.StdEncoding.EncodeToString(fixture.wave)) + `),c=>c.charCodeAt(0));const stored=await overgo.api.upload('/artifacts/intake',new File([bytes],'native-output.wav',{type:'audio/wav'}));window.authMediaURL='/artifacts/content?id='+encodeURIComponent(stored.id);const authenticated=await overgo.api.blob(authMediaURL);if(authenticated.size!==bytes.length)return false;window.authMediaCard=authMediaThread.mediaCard({kind:'audio',url:authMediaURL,artifact:stored.id,caption:'Stored audio fixture'});window.authMediaPlayer=authMediaCard.querySelector('audio');authMediaPlayer.muted=true;return true;})()`)
	settle(`!!authMediaPlayer.error || authMediaPlayer.readyState>=1`)
	for _, viewport := range webuilane.ScreenViewports {
		if findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "authenticated-audio"); err != nil || len(findings) != 0 {
			t.Fatalf("%s: findings=%v err=%v", viewport.Name, findings, err)
		}
	}
	check(`!authMediaPlayer.error && authMediaPlayer.readyState>=1`)
	check(`(async()=>{
   window.nativeAudio=document.querySelector('.chat-log audio');
   window.nativeAudioCard=nativeAudio.closest('.audio-result');
   window.nativeDownloadLink=nativeAudioCard.querySelector('.audio-download');
   const cardRect=nativeAudioCard.getBoundingClientRect(),playerRect=nativeAudio.getBoundingClientRect();
   if(playerRect.width<cardRect.width-2 || [...nativeAudioCard.querySelectorAll('details')].some(details=>details.open))return false;
   window.nativeOriginalBlob=overgo.api.blob;window.nativeDownloadPath=nativeDownloadLink.getAttribute('href');
   window.nativeExpectedBody=await nativeOriginalBlob.call(overgo.api,nativeDownloadPath);
   window.nativeOriginalClick=HTMLAnchorElement.prototype.click;window.nativeDownloads=[];
   window.nativeOriginalCreateURL=URL.createObjectURL;window.nativeDownloadBlobs=new Map();
   URL.createObjectURL=function(body){const url=nativeOriginalCreateURL.call(this,body);nativeDownloadBlobs.set(url,body);return url;};
   window.nativeOriginalRevokeURL=URL.revokeObjectURL;window.nativeRevokedURLs=new Set();
   URL.revokeObjectURL=function(url){nativeRevokedURLs.add(url);return nativeOriginalRevokeURL.call(this,url);};
   HTMLAnchorElement.prototype.click=function(){if(this.href.startsWith('blob:')&&this.download){nativeDownloads.push({name:this.download,url:this.href});return;}return nativeOriginalClick.call(this);};
   window.nativeDownloadCalls=[];
   overgo.api.blob=function(path,options){if(path!==nativeDownloadPath)return nativeOriginalBlob.call(this,path,options);return new Promise((resolve,reject)=>nativeDownloadCalls.push({resolve,reject,signal:options.signal}));};
   nativeDownloadLink.focus();return true;
 })()`)
	pressKey(t, ctx, browser, "Enter", 13)
	settle(`nativeDownloadCalls.length===1`)
	check(`(()=>{nativeDownloadCalls[0].reject(new Error('controlled download failure'));return true;})()`)
	settle(`nativeAudioCard.querySelector('.artifact-load-error')?.textContent.includes('controlled download failure')`)
	check(`(()=>{nativeAudioCard.querySelector('.artifact-load-error button').focus();return nativeDownloadLink.isConnected;})()`)
	pressKey(t, ctx, browser, "Enter", 13)
	settle(`nativeDownloadCalls.length===2`)
	check(`(()=>{nativeDownloadLink.click();return nativeDownloadCalls.length===2 && !nativeAudioCard.querySelector('.artifact-load-error') && document.activeElement===nativeDownloadLink;})()`)
	check(`(async()=>{nativeDownloadCalls[1].resolve(await nativeOriginalBlob.call(overgo.api,nativeDownloadPath));return true;})()`)
	settle(`nativeDownloads.length===1`)
	check(`(async()=>{
   if(nativeDownloads[0].name!=='audio.wav')return false;
   const received=await nativeDownloadBlobs.get(nativeDownloads[0].url).arrayBuffer();
   const expected=await nativeExpectedBody.arrayBuffer();
   const digest=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))).join(',');
   return received.byteLength===expected.byteLength && await digest(received)===await digest(expected);
 })()`)
	if err := browser.Call(ctx, "Runtime.evaluate", map[string]any{"expression": `nativeAudio.loop=true;nativeAudio.play()`, "userGesture": true, "awaitPromise": true}, nil); err != nil {
		t.Fatal(err)
	}
	settle(`!nativeAudio.paused`)
	check(`(()=>{window.dispatchEvent(new PageTransitionEvent('pagehide',{persisted:true}));return nativeAudio.paused;})()`)
	if err := browser.Call(ctx, "Runtime.evaluate", map[string]any{"expression": `nativeAudio.play()`, "userGesture": true, "awaitPromise": true}, nil); err != nil {
		t.Fatal(err)
	}
	check(`(()=>{nativeDownloadLink.click();return true;})()`)
	settle(`nativeDownloadCalls.length===3`)
	check(`(()=>{nativeAudioCard.remove();return true;})()`)
	settle(`nativeAudio.paused && nativeDownloadCalls[2].signal.aborted && nativeRevokedURLs.has(nativeAudio.src) && nativeRevokedURLs.has(nativeDownloads[0].url)`)
	check(`(async()=>{nativeDownloadCalls[2].resolve(nativeExpectedBody);await new Promise(resolve=>requestAnimationFrame(resolve));HTMLAnchorElement.prototype.click=nativeOriginalClick;URL.createObjectURL=nativeOriginalCreateURL;URL.revokeObjectURL=nativeOriginalRevokeURL;overgo.api.blob=nativeOriginalBlob;return nativeDownloads.length===1 && !document.querySelector('.artifact-load-error');})()`)
	t.Log("audio output leg: full-width native player, explicit byte-exact WAV download, failed-fetch keyboard retry, duplicate request suppression, pagehide/removal pause and cancelled late download")
}
