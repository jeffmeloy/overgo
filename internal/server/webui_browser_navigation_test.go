package server

import (
	"context"
	"errors"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

func TestWebUIBrowserConversationNavigation(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.ShortIntegration + ": conversation navigation runs through cmd/webui-lane")
	}
	h, _, _ := conversationHistoryFixture(t)
	server := httptest.NewServer(h)
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeoutCause(t.Context(), 45*time.Second, errors.New("conversation navigation did not settle"))
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
			var state string
			_ = browser.Evaluate(t.Context(), `JSON.stringify({text:document.body.innerText,errors:overgo.errors})`, &state)
			t.Fatalf("%v: %s", err, state)
		}
	}
	if err := browser.SetViewport(ctx, 390, 844); err != nil {
		t.Fatal(err)
	}
	settle(`!!document.querySelector('.composer textarea') && document.querySelectorAll('#history-rows .conversation').length===2`)
	check(`(() => {
  window.historyButton=text=>[...document.querySelectorAll('#conversation-list button')].find(node=>node.textContent===text);
  window.historySearch=text=>{const field=document.querySelector('.history-search input');field.value=text;field.form.requestSubmit();};
  window.historyRows=()=>[...document.querySelectorAll('#history-rows .conversation')];
  window.historyGet=overgo.api.get;window.historyPost=overgo.api.post;
  document.querySelector('#navigation-toggle').click();historyButton('Load older conversations').click();return true;
})()`)
	settle(`historyRows().length===4 && !historyButton('Load older conversations').disabled`)
	check(`(() => {historyButton('Load older conversations').click();return true;})()`)
	settle(`historyRows().length===5 && historyButton('Load older conversations').hidden`)
	check(`new Set(historyRows().map(row=>row.dataset.latest)).size===5 && historyRows().filter(row=>row.querySelector('.conversation-title').textContent==='Duplicate').length===2`)
	check(`(() => {historySearch('duplicate');return true;})()`)
	settle(`historyRows().length===2 && historyRows().every(row=>row.textContent.includes('Duplicate'))`)
	check(`(() => {historySearch('absent');return true;})()`)
	settle(`historyRows().length===0 && document.querySelector('#conversation-list').textContent.includes('No matching titles')`)
	check(`(() => {historySearch('Older title');return true;})()`)
	settle(`historyRows().length===1`)
	check(`(() => {historyRows()[0].querySelector('.conversation-open').focus();return true;})()`)
	pressKey(t, ctx, browser, "Enter", 13)
	settle(`document.querySelectorAll('.chat-log .msg').length===10 && !document.querySelector('.send-button').disabled && document.querySelector('#navigation-toggle').getAttribute('aria-expanded')==='false'`)
	check(`(() => {
  const input=document.querySelector('.composer textarea');input.value='Keep this unsent draft';input.dispatchEvent(new Event('input',{bubbles:true}));
  window.historySelected=overgo.conversation();window.historyTranscript=document.querySelector('.chat-log');
  document.querySelector('#navigation-toggle').click();historyRows()[0].querySelector('.history-options').click();historyRows()[0].querySelector('[aria-label="rename conversation"]').click();
  let fail=true;overgo.api.post=async function(path,body,options){if(path==='/interactions/label'&&fail){fail=false;throw new Error('Rename temporarily unavailable');}return historyPost.call(this,path,body,options);};
  document.querySelector('.history-editor input').value='Renamed history';return true;
})()`)
	pressKey(t, ctx, browser, "Enter", 13)
	settle(`!!document.querySelector('.history-editor input:not(:disabled)') && historyRows()[0].textContent.includes('Rename temporarily unavailable')`)
	check(`document.querySelector('.history-editor input').value==='Renamed history' && document.activeElement===document.querySelector('.history-editor input')`)
	// The current search no longer matches after rename; the selected transcript survives.
	pressKey(t, ctx, browser, "Enter", 13)
	settle(`historyRows().length===0 && overgo.conversation().title==='Renamed history'`)
	check(`document.querySelector('.chat-log')===historyTranscript && document.querySelector('.composer textarea').value==='Keep this unsent draft' && document.activeElement===historyButton('Archived')`)
	check(`(() => {overgo.api.post=historyPost;historySearch('Renamed history');return true;})()`)
	settle(`historyRows().length===1`)
	check(`(() => {historyRows()[0].querySelector('.history-options').click();historyRows()[0].querySelector('[aria-label="archive conversation"]').click();return true;})()`)
	settle(`historyRows().length===0 && overgo.conversation().archived`)
	check(`(() => {historyButton('Archived').click();return true;})()`)
	settle(`historyRows().length===1`)
	check(`(() => {historyRows()[0].querySelector('.history-options').click();historyRows()[0].querySelector('[aria-label="restore conversation"]').click();return true;})()`)
	settle(`historyRows().length===0 && !overgo.conversation().archived`)
	check(`document.querySelector('.chat-log')===historyTranscript && document.querySelector('.composer textarea').value==='Keep this unsent draft'`)
	check(`(() => {document.querySelector('.history-search input').value='';historyButton('Archived').click();return true;})()`)
	settle(`historyRows().length===2 && !historyButton('Load older conversations').hidden`)
	// A real intervening store commit invalidates the continuation without discarding rows.
	check(`(async () => {await historyPost.call(overgo.api,'/interactions/label',{root:historySelected.root,title:'Renamed again',archived:false});historyButton('Load older conversations').click();return true;})()`)
	settle(`document.querySelector('#conversation-list [role=alert]').textContent.includes('History changed') && !historyButton('Reload history').hidden`)
	check(`historyRows().length===2 && document.querySelector('.composer textarea').value==='Keep this unsent draft'`)
	check(`(() => {historyButton('Reload history').click();return true;})()`)
	settle(`historyButton('Reload history').hidden && historyRows().length===2`)
	check(`(() => {
  overgo.api.get=async function(path,options){if(path.startsWith('/interactions?'))throw new Error('History temporarily unavailable');return historyGet.call(this,path,options);};
  historySearch('Duplicate');return true;
})()`)
	settle(`document.querySelector('#conversation-list [role=alert]').textContent.includes('History temporarily unavailable')`)
	check(`historyRows().length===2 && !historyButton('Reload history').hidden`)
	check(`(() => {overgo.api.get=historyGet;historyButton('Reload history').click();return true;})()`)
	settle(`historyButton('Reload history').hidden && historyRows().every(row=>row.querySelector('.conversation-title').textContent==='Duplicate')`)
	// An obsolete request can finish even if a transport ignores abort; latest search wins.
	check(`(() => {
  overgo.api.get=async function(path,options){if(path.startsWith('/interactions?')&&new URL(path,location.origin).searchParams.get('q')==='older')return new Promise(resolve=>window.releaseHistory=resolve);return historyGet.call(this,path,options);};
  historySearch('older');return true;
})()`)
	settle(`typeof releaseHistory==='function'`)
	check(`(() => {historySearch('Other model');return true;})()`)
	settle(`historyRows().length===1 && historyRows()[0].dataset.latest==='resp_other'`)
	check(`(async () => {releaseHistory({conversations:[]});await Promise.resolve();await Promise.resolve();overgo.api.get=historyGet;return historyRows().length===1;})()`)
	check(`(() => {historyRows()[0].querySelector('.conversation-open').click();return true;})()`)
	settle(`document.querySelector('.chat-log').textContent.includes('An answer from another model') && document.querySelector('.send-button').disabled && document.querySelector('#panel-chat').textContent.includes('Choose model')`)
	// Failed reads remain read-only, retain the draft and offer a real remount retry.
	check(`(() => {
  let fail=true;overgo.api.get=async function(path,options){if(path.startsWith('/interactions/messages')&&fail){fail=false;throw new Error('Transcript temporarily unavailable');}return historyGet.call(this,path,options);};
  overgo.openConversation(historySelected);return true;
})()`)
	settle(`document.querySelector('.chat-log').textContent.includes('Transcript temporarily unavailable') && !![...document.querySelectorAll('#panel-chat button')].find(node=>node.textContent==='Retry loading')`)
	check(`document.querySelector('.send-button').disabled && document.querySelector('.composer textarea').readOnly && document.querySelector('.composer textarea').value==='Keep this unsent draft'`)
	check(`(() => {[...document.querySelectorAll('#panel-chat button')].find(node=>node.textContent==='Retry loading').click();return true;})()`)
	settle(`document.querySelectorAll('.chat-log .msg').length===10 && !document.querySelector('.send-button').disabled`)
	check(`document.querySelector('.composer textarea').value==='Keep this unsent draft' && !document.querySelector('.composer textarea').readOnly && overgo.errors.length===0`)
	check(`(() => {overgo.api.get=historyGet;historySearch('');return true;})()`)
	settle(`historyRows().length===2`)
	for _, viewport := range []webuilane.Viewport{{Name: "phone", Width: 390, Height: 844}, {Name: "desktop", Width: 1280, Height: 800}} {
		if err := browser.SetViewport(ctx, viewport.Width, viewport.Height); err != nil {
			t.Fatal(err)
		}
		if viewport.Name == "phone" {
			check(`(() => {document.querySelector('#navigation-toggle').click();return true;})()`)
		}
		for _, scheme := range webuilane.ColourSchemes {
			if err := browser.SetColorScheme(ctx, scheme); err != nil {
				t.Fatal(err)
			}
			check(`(async () => {await new Promise(resolve=>requestAnimationFrame(resolve));await Promise.allSettled(document.getAnimations().map(animation=>animation.finished));return true;})()`)
			if findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "conversation-history-"+scheme); err != nil || len(findings) != 0 {
				t.Fatalf("history %s %s layout: %v, %v", viewport.Name, scheme, findings, err)
			}
		}
	}
	t.Log("conversation navigation leg: real stored history paging, search, rename failure/retry, archive/restore, stale cursor, late search, cross-model transcript, failed-load draft recovery and keyboard drawer navigation passed; physical-device behavior is outside this Chromium leg")
}
