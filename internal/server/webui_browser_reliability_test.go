package server

import (
	"context"
	"errors"
	"fmt"
	"image"
	"image/png"
	"net/http"
	"net/http/httptest"
	"os"
	"sync/atomic"
	"testing"
	"time"

	"overgo/internal/overgodb"
	"overgo/internal/testevidence"
	"overgo/internal/webuilane"
)

// Exercise timing and recovery failures through the embedded client in Chromium.
func TestWebUIBrowserReliability(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testevidence.ShortIntegrationSkip + ": browser reliability runs through cmd/webui-lane")
	}
	repository, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer repository.Close()
	handler := newTestHandlerForRepository(t, repository, responseRecipeGenerator(t, &fakeGenerator{}))
	defer handler.Close()
	var started, cancelled, followed, authorized atomic.Int32
	downloaded := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/v1/responses":
			id := started.Add(1)
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprintf(w, "event: response.created\ndata: {\"response\":{\"id\":\"resp_gui_%d\"}}\n\n", id)
			w.(http.Flusher).Flush()
			<-r.Context().Done()
			cancelled.Add(1)
		case r.URL.Path == "/interactions/messages":
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"response":"resp_gui_1","messages":[{"role":"user","content":"First conversation","response":"resp_gui_1"},{"role":"assistant","content":"partial","response":"resp_gui_1"}]}`)
		case r.URL.Path == "/interactions/follow":
			followed.Add(1)
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "event: response.created\ndata: {\"response\":{\"id\":\"resp_gui_1\"}}\n\nevent: response.output_text.delta\ndata: {\"delta\":\"Recovered answer\"}\n\nevent: response.completed\ndata: {\"response\":{\"id\":\"resp_gui_1\"}}\n\n")
		case r.URL.Path == "/artifacts/content" && r.URL.Query().Get("id") == "gui-test-image":
			if r.Header.Get("Authorization") != "Bearer gui-test" {
				w.WriteHeader(http.StatusUnauthorized)
				return
			}
			if authorized.Add(1) == 2 {
				close(downloaded)
			}
			w.Header().Set("Content-Type", "image/png")
			if err := png.Encode(w, image.NewRGBA(image.Rect(0, 0, 1, 1))); err != nil {
				t.Error(err)
			}
		default:
			handler.ServeHTTP(w, r)
		}
	}))
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeoutCause(t.Context(), 45*time.Second, errors.New("GUI reliability journey did not settle"))
	defer cancel()
	browser, err := webuilane.Open(ctx, path, server.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	if err := browser.Eventually(ctx, `!!document.querySelector('#panel-chat .composer')`); err != nil {
		var page string
		_ = browser.Evaluate(t.Context(), `JSON.stringify({errors:overgo.errors,text:document.body.innerText})`, &page)
		t.Log(page)
		t.Fatal(err)
	}
	// A queued close event must not steal focus from a user who has already
	// returned to the composer. The native dialog restores its opener itself.
	assertBrowserPredicate(t, ctx, browser, `(async () => {
    const dialog=document.querySelector('#settings-dialog'),input=document.querySelector('.composer textarea');
    document.querySelector('#settings-toggle').click();
    const closed=new Promise(resolve=>dialog.addEventListener('close',resolve,{once:true}));
    dialog.close();input.focus();await closed;
    return document.activeElement===input;
  })()`)
	assertBrowserPredicate(t, ctx, browser, `(() => {
    const fence=String.fromCharCode(96).repeat(3);
    return ['c++','objective-c','python title=example','',fence].every(info => {
      const node=overgo.md(fence+info+'\n<script>alert(1)</script>\n'+fence);
      return !!node.querySelector('pre code') && !node.querySelector('script') && node.textContent.includes('<script>');
    });
  })()`)
	assertBrowserPredicate(t, ctx, browser, `(() => {
    const original=overgo.capabilities; overgo.capabilities=()=>({media:{accept:['image/png'],max_image_dimension:2,max_image_pixels:4}});
    const host=overgo.el('div',{hidden:true,id:'reliability-fixture'});document.body.append(host);
    window.submissions=[];window.fixtureComposer=overgo.composer(host,{onSubmit:(text)=>submissions.push({text,parts:fixtureComposer.attachmentParts()})});
    overgo.capabilities=original;
    fixtureComposer.input.value='Composing';fixtureComposer.input.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',isComposing:true}));
    const canvas=document.createElement('canvas');canvas.width=canvas.height=1;
    const bytes=Uint8Array.from(atob(canvas.toDataURL().split(',')[1]),c=>c.charCodeAt(0));
    fixtureComposer.addFile(new File([bytes],'one.png',{type:'image/png'}));
    fixtureComposer.input.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter'}));
    return submissions.length===0 && fixtureComposer.element.querySelector('.send-button').disabled;
  })()`)
	if err := browser.Eventually(ctx, `!fixtureComposer.element.querySelector('.send-button').disabled && !!fixtureComposer.attachments[0].dataURL`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `(() => {fixtureComposer.input.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter'}));fixtureComposer.clearAttachments();fixtureComposer.addFile(new File(['broken'],'bad.png',{type:'image/png'}));return submissions.length===1 && submissions[0].parts.length===1;})()`)
	if err := browser.Eventually(ctx, `!!fixtureComposer.attachments[0].refusal && fixtureComposer.element.querySelector('.send-button').disabled`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `(() => {
    document.querySelector('#reliability-fixture').remove();overgo.setKey('gui-test',false);
    const image=overgo.el('img',{id:'protected-image',src:'/artifacts/content?id=gui-test-image',alt:'protected preview'});
    document.querySelector('#panel-chat .chat-log').append(image);return true;
  })()`)
	if err := browser.Eventually(ctx, `document.querySelector('#protected-image').naturalWidth===1 && document.querySelector('#protected-image').src.startsWith('blob:')`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `(() => {const link=overgo.el('a',{href:'/artifacts/content?id=gui-test-image',download:'preview.png',text:'download'});document.body.append(link);link.click();return true;})()`)
	if err := browser.Eventually(ctx, `document.querySelector('#protected-image').complete`); err != nil {
		t.Fatal(err)
	}
	// The browser requests the payload again for the download, with the same key.
	select {
	case <-ctx.Done():
		t.Fatal("authenticated download did not request its bytes")
	case <-downloaded:
	}
	say(t, ctx, browser, "First conversation")
	if err := browser.Eventually(ctx, `overgo.conversation() && overgo.conversation().root==='resp_gui_1'`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `(() => {window.oldComposer=document.querySelector('#panel-chat .composer');overgo.openConversation(null);return true;})()`)
	if err := browser.Eventually(ctx, `document.querySelector('#panel-chat .composer')!==oldComposer && !!document.querySelector('#panel-chat .front-empty')`); err != nil {
		t.Fatal(err)
	}
	if followed.Load() != 0 {
		t.Fatal("a new conversation followed the old response")
	}
	assertBrowserPredicate(t, ctx, browser, `(() => {overgo.openConversation({root:'resp_gui_1',latest:'resp_gui_1'});return true;})()`)
	if err := browser.Eventually(ctx, `document.querySelector('#panel-chat .chat-log').textContent.includes('Recovered answer') && !document.querySelector('#panel-chat .send-button').disabled`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `document.querySelectorAll('#panel-chat .msg.assistant').length===1 && !document.querySelector('#panel-chat .chat-log').textContent.includes('partial') && overgo.errors.length===0`)
	assertBrowserPredicate(t, ctx, browser, `(() => {window.originalServedModel=overgo.servedModel;overgo.servedModel=()=>({model:'another-model'});overgo.openConversation({root:'resp_gui_1',latest:'resp_gui_1',model:'original-model'});return true;})()`)
	if err := browser.Eventually(ctx, `document.querySelector('#panel-chat .composer textarea').readOnly && document.querySelector('#panel-chat .send-button').disabled && document.querySelector('#panel-chat').textContent.includes('Choose its original model')`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `(() => {overgo.servedModel=originalServedModel;overgo.openConversation(null);return true;})()`)
	if err := browser.Eventually(ctx, `!!document.querySelector('#panel-chat .front-empty') && !document.querySelector('#panel-chat .send-button').disabled`); err != nil {
		t.Fatal(err)
	}
	if started.Load() != 1 || followed.Load() != 1 || cancelled.Load() != 1 {
		t.Fatalf("response ownership: started=%d followed=%d cancelled=%d", started.Load(), followed.Load(), cancelled.Load())
	}
	t.Log("reliability leg: Markdown fences terminate; IME preserves composition; attachments await decoding; protected previews/downloads authenticate; conversation switches cancel old mounts and recover only the selected chain")
}
