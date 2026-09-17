package server

import (
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

func TestWebUIBrowserTurnLimit(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.ShortIntegration + ": turn limit runs through cmd/webui-lane")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	handler := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{pieces: []string{"Partial answer", " continued"}}))
	server := httptest.NewServer(handler)
	defer func() { _ = handler.Close(); server.Close() }()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	browser, err := webuilane.Open(t.Context(), path, server.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	settle := func(predicate string) {
		t.Helper()
		if err := browser.Eventually(t.Context(), predicate); err != nil {
			var page string
			_ = browser.Evaluate(t.Context(), "document.body.innerText", &page)
			t.Fatalf("%v; page: %s", err, page)
		}
	}
	check := func(predicate string) { t.Helper(); assertBrowserPredicate(t, t.Context(), browser, predicate) }
	settle(`!!document.querySelector('#panel-chat .composer')`)
	check(`(() => { document.querySelector('#conversation-settings input[aria-label="max tokens"]').value='1'; return true; })()`)
	say(t, t.Context(), browser, "Give a bounded answer")
	settle(`!document.querySelector('.send-button').disabled && document.querySelector('.chat-log').textContent.includes('Output limit reached')`)
	check(`!document.querySelector('.msg.error') && !sessionStorage.getItem('overgo.inflight') && document.querySelector('.msg.assistant .body').textContent==='Partial answer'`)
	check(`(async () => {
    window.limitSelection=overgo.conversation();
    const history=await overgo.api.get('/interactions/messages?response='+limitSelection.latest);
    if(history.status!=='incomplete' || history.incomplete_details.reason!=='max_output_tokens') return false;
    const follow=await overgo.api.stream('/interactions/follow?response='+limitSelection.latest,null,{method:'GET'});
    const events=[]; for await(const event of overgo.streams.responses(follow)) events.push(event);
    return events.filter(event=>event.type==='done').length===1 && events.at(-1).status==='incomplete';
  })()`)
	check(`(() => {window.limitComposer=document.querySelector('.composer');overgo.openConversation(limitSelection);return true;})()`)
	settle(`!!document.querySelector('.composer') && document.querySelector('.composer')!==limitComposer && !document.querySelector('.send-button').disabled && document.querySelector('.chat-log').textContent.includes('Output limit reached')`)
	check(`!document.querySelector('.msg.error') && document.querySelectorAll('.msg.assistant').length===1 && !sessionStorage.getItem('overgo.inflight')`)
	check(`(() => {overgo.inspectTurn(limitSelection.latest);return true;})()`)
	settle(`!document.querySelector('#inspector').hidden && document.querySelector('#inspector').textContent.includes('Output limit reached')`)
	check(`(() => {const inspector=document.querySelector('#inspector');const valid=inspector.textContent.includes('Partial answer saved');inspector.querySelector('[aria-label="close the inspector"]').click();return valid;})()`)
	check(`(() => {document.querySelector('#conversation-settings input[aria-label="max tokens"]').value='3';return true;})()`)
	say(t, t.Context(), browser, "Continue the answer")
	settle(`!document.querySelector('.send-button').disabled && document.querySelectorAll('.msg.assistant').length===2`)
	check(`(async () => { const history=await overgo.api.get('/interactions/messages?response='+overgo.conversation().latest);return history.status==='completed' && !history.incomplete_details && history.previous===limitSelection.latest && !document.querySelector('.msg.error') && !sessionStorage.getItem('overgo.inflight') && overgo.errors.length===0; })()`)
	check(`(() => {window.limitComposer=document.querySelector('.composer');overgo.openConversation(overgo.conversation());return true;})()`)
	settle(`!!document.querySelector('.composer') && document.querySelector('.composer')!==limitComposer && !document.querySelector('.send-button').disabled && document.querySelectorAll('.msg.assistant').length===2`)
	check(`[...document.querySelectorAll('.msg.assistant .tag')].filter(tag=>tag.textContent==='Output limit reached').length===1 && document.querySelector('.msg.assistant .role').textContent.includes('Output limit reached')`)
	t.Log("turn limit leg: real Responses generation with synthetic tokens preserves partial output and limit reason through stream, follow, reopen and continuation; no false connection-ended error or stale recovery handle")
}
