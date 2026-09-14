package server

import (
	"context"
	"encoding/base64"
	"errors"
	"net/http/httptest"
	"os"
	"slices"
	"strconv"
	"testing"
	"time"

	"overgo/internal/testevidence"
	"overgo/internal/webuilane"
)

func TestWebUIBrowserTranscriptionCleanup(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testevidence.ShortIntegrationSkip + ": transcription cleanup runs through cmd/webui-lane")
	}
	fixture := newTranscriptionHTTPFixture(t, nil)
	fixture.handler.config.AudioProjector = &fakeAudioProjector{}
	fixture.handler.config.APIKey = ""
	server := httptest.NewServer(fixture.handler)
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeoutCause(t.Context(), 90*time.Second, errors.New("transcription cleanup did not settle"))
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
	settle(`!!document.querySelector('.composer textarea')`)
	// Hold only the transcription reply to deliver it after invalidation. The
	// selection journey separately exercises the actual native multipart route.
	check(`(() => {
 const create=overgo.composer;overgo.composer=(...args)=>(window.cleanupOwned=create(...args));
 const form=overgo.api.form;window.cleanupCalls=[];
 overgo.api.form=function(path,body,options){if(path!=='/v1/audio/transcriptions')return form.call(this,path,body,options);return new Promise(resolve=>cleanupCalls.push({signal:options.signal,resolve}));};
 window.cleanupWave=Uint8Array.from(atob(` + strconv.Quote(base64.StdEncoding.EncodeToString(fixture.wave)) + `),c=>c.charCodeAt(0));
 return true;
})()`)
	cases := []struct{ name, action string }{
		{"mode", `cleanupOwned.element.querySelector('[aria-label="mode"]').focus();`},
		{"read-only", `cleanupOwned.setReadOnly(true);`},
		{"pagehide", `window.dispatchEvent(new PageTransitionEvent('pagehide',{persisted:true}));`},
		{"clear", `cleanupOwned.clearAttachments();`},
		{"remove", `cleanupOwned.element.querySelector('.attachment-row button[aria-label^="Remove "]').click();`},
		{"dispose", `cleanupOwned.dispose();`},
		{"retry", `cleanupOwned.element.querySelector('.attachment-row button[aria-label^="Transcribe "]').focus();`},
	}
	for _, stage := range []string{"pending", "offer"} {
		for _, test := range cases {
			if stage == "pending" && test.name == "retry" {
				continue
			}
			t.Log(stage + "/" + test.name)
			check(`(() => {window.cleanupOld=document.querySelector('.composer');overgo.openConversation(null);return true;})()`)
			settle(`!!window.cleanupOwned && cleanupOwned.element.isConnected && cleanupOwned.element!==cleanupOld`)
			check(`(() => {cleanupOwned.input.value='Keep this draft';cleanupOwned.input.dispatchEvent(new Event('input'));cleanupOwned.addFile(new File([cleanupWave],'cleanup.wav',{type:'audio/wav'}));return true;})()`)
			settle(`!!cleanupOwned.element.querySelector('.attachment-row button[aria-label^="Transcribe "]:not([hidden]):not([disabled])')`)
			check(`(() => {window.cleanupCount=cleanupCalls.length;cleanupOwned.element.querySelector('.attachment-row button[aria-label^="Transcribe "]').focus();return true;})()`)
			pressKey(t, ctx, browser, "Enter", 13)
			settle(`cleanupCalls.length===cleanupCount+1`)
			if stage == "offer" {
				check(`(() => {cleanupCalls.at(-1).resolve({text:'Do not insert stale words'});return true;})()`)
				settle(`!!cleanupOwned.element.querySelector('.transcript-offer:not([hidden]) button')`)
				check(`(() => {window.cleanupOldInsert=cleanupOwned.element.querySelector('.transcript-offer button');return true;})()`)
			}
			playback := stage == "pending" && slices.Contains([]string{"pagehide", "dispose", "remove", "clear"}, test.name)
			if playback {
				check(`(async () => {window.cleanupPlayer=cleanupOwned.element.querySelector('audio');cleanupPlayer.muted=true;cleanupPlayer.loop=true;await cleanupPlayer.play();return !cleanupPlayer.paused;})()`)
			}
			check(`(() => {` + test.action + `return true;})()`)
			if test.name == "retry" {
				pressKey(t, ctx, browser, "Enter", 13)
			}
			if playback {
				check(`cleanupPlayer.paused`)
			}
			if test.name == "mode" {
				var delta int
				if err := browser.Evaluate(ctx, `(() => {const mode=cleanupOwned.element.querySelector('[aria-label="mode"]');return [...mode.options].findIndex(option=>option.value==='transcription')-mode.selectedIndex;})()`, &delta); err != nil {
					t.Fatal(err)
				}
				for delta != 0 {
					if delta > 0 {
						pressKey(t, ctx, browser, "ArrowDown", 40)
						delta--
					} else {
						pressKey(t, ctx, browser, "ArrowUp", 38)
						delta++
					}
				}
				check(`cleanupOwned.mode()==='transcription'`)
			}
			if stage == "pending" {
				check(`cleanupCalls.at(-1).signal.aborted`)
				check(`(() => {cleanupCalls.at(-1).resolve({text:'Do not insert stale words'});return true;})()`)
			} else {
				check(`(() => {cleanupOldInsert.click();return true;})()`)
			}
			check(`cleanupOwned.input.value==='Keep this draft' && !cleanupOwned.element.querySelector('.transcript-offer:not([hidden])')`)
			if test.name == "read-only" {
				check(`cleanupOwned.element.querySelector('.send-button').disabled && [...cleanupOwned.element.querySelectorAll('.attachment-actions button')].every(button=>button.hidden||button.disabled)`)
			}
			if stage == "pending" && (test.name == "read-only" || test.name == "pagehide") {
				for _, viewport := range slices.Concat(webuilane.ScreenViewports, []webuilane.Viewport{{Name: "phone-keyboard", Width: 390, Height: 380}}) {
					if _, err := webuilane.CaptureState(ctx, browser, "", viewport, "cleanup-resize"); err != nil {
						t.Fatal(err)
					}
					var wheel struct{ X, Y, Delta float64 }
					if err := browser.Evaluate(ctx, `(() => {const r=cleanupOwned.extras.getBoundingClientRect(),status=cleanupOwned.element.querySelector('.attachment-row [role=status]').getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2,delta:Math.ceil(Math.max(0,status.bottom-r.bottom))};})()`, &wheel); err != nil {
						t.Fatal(err)
					}
					if wheel.Delta > 0 {
						if err := browser.Call(ctx, "Input.dispatchMouseEvent", map[string]any{"type": "mouseWheel", "x": wheel.X, "y": wheel.Y, "deltaX": 0, "deltaY": wheel.Delta}, nil); err != nil {
							t.Fatal(err)
						}
					}
					settle(`(() => {const outer=cleanupOwned.extras.getBoundingClientRect(),status=cleanupOwned.element.querySelector('.attachment-row [role=status]').getBoundingClientRect();return status.top>=outer.top&&Math.ceil(status.bottom)<=Math.ceil(outer.bottom);})()`)
					if findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "transcription-"+test.name); err != nil || len(findings) != 0 {
						t.Fatalf("%s: findings=%v err=%v", viewport.Name, findings, err)
					}
					check(`(() => {const button=cleanupOwned.element.querySelector('.send-button'),r=button.getBoundingClientRect();return r.width>0&&r.height>0&&r.top>=0&&r.bottom<=innerHeight&&document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)===button;})()`)
				}
				check(`(() => {cleanupOwned.setReadOnly(false);window.dispatchEvent(new PageTransitionEvent('pageshow',{persisted:true}));return true;})()`)
				settle(`!!cleanupOwned.element.querySelector('.attachment-actions button[aria-label^="Transcribe "]:not([hidden]):not([disabled])')`)
				check(`(() => {window.cleanupRetryCount=cleanupCalls.length;const button=cleanupOwned.element.querySelector('.attachment-actions button[aria-label^="Transcribe "]');button.focus();return document.activeElement===button;})()`)
				pressKey(t, ctx, browser, "Enter", 13)
				settle(`cleanupCalls.length===cleanupRetryCount+1`)
				check(`(() => {cleanupCalls.at(-1).resolve({text:'Recovered words'});return true;})()`)
				settle(`!!cleanupOwned.element.querySelector('.transcript-offer:not([hidden]) button')`)
				check(`(() => {cleanupOwned.element.querySelector('.transcript-offer button').focus();return true;})()`)
				pressKey(t, ctx, browser, "Enter", 13)
				check(`cleanupOwned.input.value==='Keep this draft\nRecovered words' && cleanupOwned.attachments.length===1`)
			}
		}
	}
	// Reordering a refreshed catalog preserves the selected recipe. Retirement
	// requires explicit reselection; the alternate is a catalog-only fixture.
	check(`(() => {window.cleanupOld=document.querySelector('.composer');overgo.openConversation(null);return true;})()`)
	settle(`cleanupOwned.element.isConnected && cleanupOwned.element!==cleanupOld`)
	check(`(() => {cleanupOwned.input.value='Keep selection draft';cleanupOwned.addFile(new File([cleanupWave],'cleanup.wav',{type:'audio/wav'}));return true;})()`)
	settle(`!!cleanupOwned.element.querySelector('.attachment-row button[aria-label^="Transcribe "]:not([hidden]):not([disabled])')`)
	check(`(() => {
 window.cleanupChosen=cleanupOwned.element.querySelector('[aria-label="Transcription model"]').value;
 window.cleanupCatalog='reordered';window.cleanupRefreshes=0;const get=overgo.api.get;
 overgo.api.get=async function(path,options){const result=await get.call(this,path,options);if(path!=='/generation/capabilities')return result;cleanupRefreshes++;const other={...result[0],recipe:'fixture-other-asr',name:'Other transcription model'};return cleanupCatalog==='retired'?[other]:[other,...result];};
 window.dispatchEvent(new PageTransitionEvent('pagehide',{persisted:true}));window.dispatchEvent(new PageTransitionEvent('pageshow',{persisted:true}));return true;
})()`)
	settle(`cleanupRefreshes===1 && !cleanupOwned.element.querySelector('[aria-label="Transcription model"]').disabled`)
	check(`cleanupOwned.element.querySelector('[aria-label="Transcription model"]').value===cleanupChosen`)
	check(`(() => {cleanupCatalog='retired';window.dispatchEvent(new PageTransitionEvent('pagehide',{persisted:true}));window.dispatchEvent(new PageTransitionEvent('pageshow',{persisted:true}));return true;})()`)
	settle(`cleanupRefreshes===2 && cleanupOwned.element.querySelector('.transcription-controls [role=status]').textContent.includes('unavailable')`)
	check(`cleanupOwned.element.querySelector('.attachment-actions button[aria-label^="Transcribe "]').disabled && !cleanupOwned.element.querySelector('[aria-label="Transcription model"]').disabled && cleanupOwned.element.querySelector('[aria-label="Transcription model"]').value===cleanupChosen`)
	for _, viewport := range webuilane.ScreenViewports {
		if findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "transcription-model-retired"); err != nil || len(findings) != 0 {
			t.Fatalf("%s: findings=%v err=%v", viewport.Name, findings, err)
		}
		check(`(() => {const picker=cleanupOwned.element.querySelector('[aria-label="Transcription model"]');return picker.parentElement.getBoundingClientRect().height<=picker.getBoundingClientRect().height;})()`)
	}
	check(`(() => {cleanupOwned.element.querySelector('[aria-label="Transcription model"]').focus();return true;})()`)
	pressKey(t, ctx, browser, "ArrowDown", 40)
	check(`cleanupOwned.element.querySelector('[aria-label="Transcription model"]').value==='fixture-other-asr' && !cleanupOwned.element.querySelector('.attachment-actions button[aria-label^="Transcribe "]').disabled && cleanupOwned.input.value==='Keep selection draft'`)
	check(`overgo.errors.length===0`)
	t.Log("transcription cleanup leg: mode/read-only/pagehide/clear/remove/dispose abort pending work and invalidate ready callbacks while retaining drafts; synthetic lifecycle and controlled reply timing")
}
