package server

import (
	"context"
	"errors"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"overgo/internal/overgodb"
	"overgo/internal/testevidence"
	"overgo/internal/webuilane"
)

func TestWebUIBrowserConversationControl(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testevidence.ShortIntegrationSkip + ": conversation control runs through cmd/webui-lane")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	generator := &responseControlGenerator{
		recipeInspectorGenerator: responseRecipeGenerator(t, &fakeGenerator{pieces: []string{"Recovered answer"}}),
		starts:                   make(chan context.Context), release: make(chan error),
	}
	handler := newTestHandlerForRepository(t, store, generator)
	server := httptest.NewServer(handler)
	defer func() { _ = handler.Close(); server.Close() }()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeoutCause(t.Context(), 45*time.Second, errors.New("conversation control journey did not settle"))
	defer cancel()
	browser, err := webuilane.Open(ctx, path, server.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	settle := func(predicate string) {
		t.Helper()
		if err := browser.Eventually(ctx, predicate); err != nil {
			var page string
			_ = browser.Evaluate(t.Context(), "document.body.innerText", &page)
			t.Fatalf("%v; page: %s", err, page)
		}
	}
	next := func() context.Context {
		t.Helper()
		select {
		case execution := <-generator.starts:
			return execution
		case <-ctx.Done():
			var page string
			_ = browser.Evaluate(t.Context(), "document.body.innerText", &page)
			t.Fatalf("generation did not start: %s", page)
			return nil
		}
	}
	newChat := func() {
		t.Helper()
		assertBrowserPredicate(t, ctx, browser, `(() => {window.controlOldComposer=document.querySelector('#panel-chat .composer');overgo.openConversation(null);return true;})()`)
		settle(`document.querySelector('#panel-chat .composer')!==controlOldComposer && !!document.querySelector('#panel-chat .front-empty')`)
	}
	send := func(prompt string) {
		t.Helper()
		assertBrowserPredicate(t, ctx, browser, `(() => {document.querySelector('#conversation-settings input[aria-label="max tokens"]').value='2';return true;})()`)
		say(t, ctx, browser, prompt)
	}
	settle(`!!document.querySelector('#panel-chat .composer')`)
	// EOF is not completion, and an error cannot be undone by a later frame.
	assertBrowserPredicate(t, ctx, browser, `(async () => {
    const frame=(event,body)=>'event: '+event+'\ndata: '+JSON.stringify(body)+'\n\n';
    let observed=[],interrupted=false;
    try {for await(const event of overgo.streams.responses(new Response(frame('response.created',{response:{id:'truncated'}})))) observed.push(event);}
    catch (_) {interrupted=true;}
    if(!interrupted || observed.some(event=>event.type==='done')) return false;
    observed=[];
    for await(const event of overgo.streams.responses(new Response(frame('response.failed',{delta:'failed'})+frame('response.completed',{response:{id:'wrong'}})))) observed.push(event);
    return observed.length===1 && observed[0].status==='failed';
  })()`)

	send("Stop this response")
	stopped := next()
	settle(`!!overgo.conversation() && !!sessionStorage.getItem('overgo.inflight')`)
	assertBrowserPredicate(t, ctx, browser, `(() => {window.stoppedSelection=overgo.conversation();document.querySelector('.stop-button').click();return true;})()`)
	select {
	case <-stopped.Done():
	case <-ctx.Done():
		t.Fatal("Stop did not cancel the generator")
	}
	settle(`!document.querySelector('.send-button').disabled && document.querySelector('.chat-log').textContent.includes('Stopped')`)
	assertBrowserPredicate(t, ctx, browser, `(async () => {
    const result=await overgo.api.post('/interactions/cancel',{response:stoppedSelection.latest,model:'test-model'});
    return result.status==='cancelled' && !sessionStorage.getItem('overgo.inflight');
  })()`)
	assertBrowserPredicate(t, ctx, browser, `(() => {window.controlOldComposer=document.querySelector('.composer');overgo.openConversation(stoppedSelection);return true;})()`)
	settle(`!!document.querySelector('.composer') && document.querySelector('.composer')!==controlOldComposer && !document.querySelector('.send-button').disabled`)
	assertBrowserPredicate(t, ctx, browser, `!document.querySelector('.msg.error') && document.querySelector('.chat-log').textContent.includes('Stopped')`)

	newChat()
	send("Recover this response")
	recovering := next()
	settle(`!!sessionStorage.getItem('overgo.inflight')`)
	assertBrowserPredicate(t, ctx, browser, `(() => {window.recoverySelection=overgo.conversation();sessionStorage.removeItem('overgo.inflight');return true;})()`)
	newChat()
	select {
	case <-recovering.Done():
		t.Fatal("navigation cancelled stored execution")
	default:
	}
	assertBrowserPredicate(t, ctx, browser, `(() => {overgo.openConversation(recoverySelection);return true;})()`)
	settle(`!!sessionStorage.getItem('overgo.inflight') && !document.querySelector('.stop-button').hidden`)
	// Reopening followed the server's in-progress record with no storage handle.
	select {
	case generator.release <- nil:
	case <-ctx.Done():
		t.Fatal("recovered execution did not wait for release")
	}
	settle(`!document.querySelector('.send-button').disabled && document.querySelector('.chat-log').textContent.includes('Recovered answer')`)
	assertBrowserPredicate(t, ctx, browser, `document.querySelectorAll('.msg.assistant').length===1 && document.querySelector('.msg.assistant .body').textContent==='Recovered answer'`)
	if generator.calls.Load() != 2 {
		t.Fatal("resume started duplicate generation")
	}

	newChat()
	send("Keep this failed prompt")
	next()
	select {
	case generator.release <- errors.New("controlled generation failure"):
	case <-ctx.Done():
		t.Fatal("failure fixture did not reach generation")
	}
	settle(`!document.querySelector('.send-button').disabled && document.querySelector('.composer textarea').value==='Keep this failed prompt' && document.querySelector('.chat-log').textContent.includes('controlled generation failure')`)
	if generator.calls.Load() != 3 {
		t.Fatal("failed prompt was retried automatically")
	}

	newChat()
	// Delay the browser's receipt of real response headers, then press Stop.
	// The cancellation must wait for the ID and target that execution.
	assertBrowserPredicate(t, ctx, browser, `(() => {
    window.controlStream=overgo.api.stream;
    overgo.api.stream=async function(path,body,options) {
      const response=await controlStream.call(this,path,body,options);
      if(path==='/v1/responses') await new Promise(resolve=>window.releaseControlHeaders=resolve);
      return response;
    };return true;
  })()`)
	send("Stop before the response ID arrives")
	early := next()
	settle(`typeof releaseControlHeaders==='function'`)
	assertBrowserPredicate(t, ctx, browser, `(() => {document.querySelector('.stop-button').click();return document.querySelector('.stop-button').textContent==='Stopping…';})()`)
	select {
	case <-early.Done():
		t.Fatal("execution cancelled without its response ID")
	default:
	}
	newChat()
	assertBrowserPredicate(t, ctx, browser, `(() => {overgo.api.stream=controlStream;releaseControlHeaders();return true;})()`)
	select {
	case <-early.Done():
	case <-ctx.Done():
		t.Fatal("queued Stop did not reach the generator")
	}
	settle(`!document.querySelector('.send-button').disabled && !!document.querySelector('#panel-chat .front-empty')`)
	if generator.calls.Load() != 4 {
		t.Fatal("early Stop started or retried another request")
	}

	// An expired handle fails visibly; it must never produce a completed answer.
	assertBrowserPredicate(t, ctx, browser, `(() => {
    sessionStorage.setItem('overgo.inflight',JSON.stringify({root:'resp_missing',response:'resp_missing',model:'test-model'}));
    overgo.openConversation({root:'resp_missing',latest:'resp_missing'});return true;
  })()`)
	settle(`document.querySelector('.send-button').disabled && document.querySelector('.composer textarea').readOnly && document.querySelector('.chat-log').textContent.includes('not found') && [...document.querySelectorAll('.chat-log button')].some(button=>button.textContent==='Retry loading')`)
	assertBrowserPredicate(t, ctx, browser, `!sessionStorage.getItem('overgo.inflight') && overgo.errors.length===0`)
	if generator.calls.Load() != 4 {
		t.Fatal("an expired response generated a replacement automatically")
	}
	t.Log("conversation control leg: real stored generation cancels on Stop and early Stop; navigation preserves execution; server-owned resume produces one answer; failures preserve prompts; expired and truncated responses never complete or retry automatically")
}
