package server

import (
	"context"
	"errors"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"overgo/internal/overgodb"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserConversationLayout exercises the work the user does, including
// a phone viewport reduced by its keyboard, instead of scoring header density.
func TestWebUIBrowserConversationLayout(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.ShortIntegration + ": browser layout runs through cmd/webui-lane")
	}
	repository, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer repository.Close()
	blocking := &fakeGenerator{started: make(chan struct{}), release: make(chan struct{})}
	defer close(blocking.release)
	handler := newTestHandlerForRepository(t, repository, responseRecipeGenerator(t, blocking))
	defer handler.Close()
	server := httptest.NewServer(handler)
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeoutCause(t.Context(), 45*time.Second, errors.New("conversation layout did not settle"))
	defer cancel()
	browser, err := webuilane.Open(ctx, path, server.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	if err := browser.Eventually(ctx, `!!document.querySelector('#panel-chat.active .composer textarea')`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `document.querySelector('#global-operation-shell').hidden && !document.querySelector('#settings-dialog').open && document.querySelector('#api-key').getBoundingClientRect().width===0`)
	for _, viewport := range []webuilane.Viewport{{Name: "desktop", Width: 1280, Height: 900}, {Name: "phone", Width: 390, Height: 844}, {Name: "phone-keyboard", Width: 390, Height: 380}, {Name: "small-phone", Width: 320, Height: 568}} {
		if err := browser.SetViewport(ctx, viewport.Width, viewport.Height); err != nil {
			t.Fatal(err)
		}
		if err := browser.Eventually(ctx, `Math.abs(document.querySelector('.shell').getBoundingClientRect().height-innerHeight)<2`); err != nil {
			t.Fatal(err)
		}
		assertBrowserPredicate(t, ctx, browser, `(() => {
    const visible = node => {const r=node.getBoundingClientRect();return r.width>0 && r.height>0 && r.left>=0 && r.right<=innerWidth && r.top>=0 && r.bottom<=innerHeight;};
    return visible(document.querySelector('.composer textarea')) && visible(document.querySelector('.send-button')) && document.querySelector('#panel-chat .chat-log').clientHeight > document.querySelector('.composer textarea').clientHeight && document.documentElement.scrollWidth<=innerWidth;
  })()`)
		if viewport.Width <= 860 {
			assertBrowserPredicate(t, ctx, browser, `getComputedStyle(document.querySelector('#navigation')).display==='none' && document.querySelector('#navigation').inert`)
			assertBrowserPredicate(t, ctx, browser, `(() => {document.querySelector('#navigation-toggle').click();return document.activeElement.id==='navigation-close' && document.querySelector('.content').inert;})()`)
			pressKey(t, ctx, browser, "Escape", 27)
			assertBrowserPredicate(t, ctx, browser, `document.activeElement.id==='navigation-toggle' && !document.querySelector('.content').inert && document.querySelector('#navigation').inert`)
		}
		assertBrowserPredicate(t, ctx, browser, `(() => {document.querySelector('#settings-toggle').click();return document.querySelector('#settings-dialog').open && document.querySelector('#api-key').getBoundingClientRect().width>0 && !!document.querySelector('#settings-dialog textarea[placeholder="system prompt (optional)"]');})()`)
		pressKey(t, ctx, browser, "Escape", 27)
		if err := browser.Eventually(ctx, `!document.querySelector('#settings-dialog').open && document.activeElement.id==='settings-toggle'`); err != nil {
			t.Fatal(err)
		}
		if dir := os.Getenv("OVERGO_WEBUI_LANE_SCREENS"); dir != "" {
			if _, err := webuilane.CaptureState(ctx, browser, dir, viewport, "conversation-redesign"); err != nil {
				t.Fatal(err)
			}
		}
	}
	// Use the production thread renderer and fill its real scrolling region.
	assertBrowserPredicate(t, ctx, browser, `(() => {
   const host=document.querySelector('#panel-chat'); const existing=host.querySelector('.chat-log');
   const thread=overgo.thread(host); existing.replaceWith(thread.node); window.layoutThread=thread;
   thread.add('assistant',Array(120).fill('A long response that leaves the composer within reach.').join('\n'));
   return thread.node.scrollHeight>thread.node.clientHeight && Math.ceil(thread.node.scrollTop+thread.node.clientHeight)>=thread.node.scrollHeight && document.querySelector('.send-button').getBoundingClientRect().bottom<=innerHeight;
 })()`)

	// Resizing after a message exists must preserve following after the shell
	// applies the visual viewport height, not scroll against the old layout.
	for _, viewport := range []webuilane.Viewport{{Name: "reply-desktop", Width: 1280, Height: 900}, {Name: "reply-phone", Width: 390, Height: 844}, {Name: "reply-keyboard", Width: 390, Height: 380}} {
		if err := browser.SetViewport(ctx, viewport.Width, viewport.Height); err != nil {
			t.Fatal(err)
		}
		if err := browser.Eventually(ctx, `Math.abs(document.querySelector('.shell').getBoundingClientRect().height-innerHeight)<2`); err != nil {
			t.Fatal(err)
		}
		assertBrowserPredicate(t, ctx, browser, `(async()=>{await new Promise(requestAnimationFrame);await new Promise(requestAnimationFrame);return true;})()`)
		assertBrowserPredicate(t, ctx, browser, `(()=>{const log=layoutThread.node;return Math.ceil(log.scrollTop+log.clientHeight)>=log.scrollHeight;})()`)
	}
	// Opening details changes the message area's height without a viewport event.
	assertBrowserPredicate(t, ctx, browser, `(()=>{window.layoutDetails=document.createElement('details');const summary=document.createElement('summary');summary.textContent='Attachment details';const content=document.createElement('p');content.textContent='A stored document retains its source identity. Its details can expand without hiding the latest reply above the composer.';layoutDetails.append(summary,content);document.querySelector('.composer-extras').append(layoutDetails);window.layoutPriorHeight=layoutThread.node.clientHeight;summary.focus();return true;})()`)
	pressKey(t, ctx, browser, "Enter", 13)
	if err := browser.Eventually(ctx, `layoutDetails.open&&layoutThread.node.clientHeight<layoutPriorHeight&&Math.ceil(layoutThread.node.scrollTop+layoutThread.node.clientHeight)>=layoutThread.node.scrollHeight`); err != nil {
		t.Fatal(err)
	}
	if findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), webuilane.Viewport{Name: "phone-keyboard", Width: 390, Height: 380}, "conversation-details-expanded"); err != nil || len(findings) != 0 {
		t.Fatalf("details expansion: %v %v", findings, err)
	}
	// Reading earlier text must not be interrupted by the next update.
	assertBrowserPredicate(t, ctx, browser, `(() => {window.layoutThread.node.scrollTop=0;return true;})()`)
	if err := browser.Eventually(ctx, `window.layoutThread.node.scrollTop===0`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `(() => {window.layoutThread.node.dispatchEvent(new Event('scroll'));window.layoutThread.add('assistant','A later update');return window.layoutThread.node.scrollTop===0;})()`)

	for _, height := range []int{844, 380} {
		if err := browser.SetViewport(ctx, 390, height); err != nil {
			t.Fatal(err)
		}
		assertBrowserPredicate(t, ctx, browser, `(async()=>{await new Promise(requestAnimationFrame);await new Promise(requestAnimationFrame);return layoutThread.node.scrollTop===0 && !document.querySelector('.jump-latest:last-of-type')?.hidden;})()`)
	}
	assertBrowserPredicate(t, ctx, browser, `(()=>{layoutDetails.querySelector('summary').focus();return true;})()`)
	pressKey(t, ctx, browser, "Enter", 13)
	assertBrowserPredicate(t, ctx, browser, `(async()=>{await new Promise(requestAnimationFrame);await new Promise(requestAnimationFrame);return !layoutDetails.open&&layoutThread.node.scrollTop===0;})()`)
	assertBrowserPredicate(t, ctx, browser, `(()=>{layoutDetails.remove();return true;})()`)
	assertBrowserPredicate(t, ctx, browser, `(() => {const thread=window.layoutThread;thread.add('user','A new message returns me to the latest reply');return Math.ceil(thread.node.scrollTop+thread.node.clientHeight)>=thread.node.scrollHeight;})()`)

	// A short conversation must start following when it becomes scrollable.
	if err := browser.SetViewport(ctx, 390, 844); err != nil {
		t.Fatal(err)
	}
	if err := browser.Eventually(ctx, `Math.abs(document.querySelector('.shell').getBoundingClientRect().height-innerHeight)<2`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `(()=>{layoutThread.reset();layoutThread.add('user','A short question');layoutThread.add('assistant','A short answer remains visible.');return true;})()`)
	var reduced int
	if err := browser.Evaluate(ctx, `(()=>{const log=layoutThread.node;const occupied=log.lastElementChild.getBoundingClientRect().bottom-log.firstElementChild.getBoundingClientRect().top;return Math.ceil(innerHeight-log.clientHeight+occupied/2)})()`, &reduced); err != nil {
		t.Fatal(err)
	}
	if err := browser.SetViewport(ctx, 390, reduced); err != nil {
		t.Fatal(err)
	}
	if err := browser.Eventually(ctx, `Math.abs(document.querySelector('.shell').getBoundingClientRect().height-innerHeight)<2`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `(async()=>{await new Promise(requestAnimationFrame);await new Promise(requestAnimationFrame);const log=layoutThread.node;return log.scrollHeight>log.clientHeight&&Math.ceil(log.scrollTop+log.clientHeight)>=log.scrollHeight;})()`)
	if err := browser.SetViewport(ctx, 390, 380); err != nil {
		t.Fatal(err)
	}
	say(t, ctx, browser, "Keep this request running until I stop it")
	if err := browser.Eventually(ctx, `document.querySelector('.send-button').disabled && !document.querySelector('.stop-button').hidden`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `(() => {const stop=document.querySelector('.stop-button'),r=stop.getBoundingClientRect(); if(r.bottom>innerHeight || r.right>innerWidth || r.top<0)return false;stop.click();return true;})()`)
	if err := browser.Eventually(ctx, `!document.querySelector('.send-button').disabled && document.querySelector('.stop-button').hidden`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `(() => {overgo.localOperation({id:'layout-operation',task:'test operation',state:'running'});return !document.querySelector('#global-operation-shell').hidden;})()`)
	assertBrowserPredicate(t, ctx, browser, `(() => {overgo.localOperation({id:'layout-operation',task:'test operation',state:'completed'});return document.getElementById('global-operation-shell').hidden && !document.querySelector('.activity-dialog').open;})()`)
	assertBrowserPredicate(t, ctx, browser, `window.overgo.errors.length===0`)

	// Replace actual conversations and verify their observed nodes are released.
	assertBrowserPredicate(t, ctx, browser, `(()=>{layoutThread.dispose();window.resizeNativeObserver=window.ResizeObserver;window.resizeOwned=[];window.ResizeObserver=class extends resizeNativeObserver{constructor(callback){super(callback);this.targets=new Set();resizeOwned.push(this)}observe(target,options){this.targets.add(target);super.observe(target,options)}disconnect(){this.targets.clear();super.disconnect()}};return true;})()`)
	for range 2 {
		assertBrowserPredicate(t, ctx, browser, `(()=>{window.resizePreviousComposer=document.querySelector('.composer');overgo.openConversation(null);return true;})()`)
		if err := browser.Eventually(ctx, `!!document.querySelector('.composer')&&document.querySelector('.composer')!==resizePreviousComposer&&resizeOwned.length>0`); err != nil {
			t.Fatal(err)
		}
		assertBrowserPredicate(t, ctx, browser, `resizeOwned.every(observer=>[...observer.targets].every(node=>node.isConnected))`)
	}
	assertBrowserPredicate(t, ctx, browser, `(()=>{window.ResizeObserver=resizeNativeObserver;return overgo.errors.length===0;})()`)
	t.Log("conversation resize leg: latest reply follows settled viewport geometry; deliberate reader position survives resizing and subsequent updates")
	t.Log("conversation layout leg: composer reachable at desktop, phone and keyboard heights; drawer and settings keyboard paths; long reply scrolling; visible stop; relevant operations")
}
