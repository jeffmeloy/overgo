package server

import (
	"context"
	"errors"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"overgo/internal/overgodb"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

func TestWebUIBrowserMobileInteraction(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.ShortIntegration + ": mobile interaction runs through cmd/webui-lane")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	generator := &responseControlGenerator{recipeInspectorGenerator: responseRecipeGenerator(t, &fakeGenerator{pieces: []string{"Final response."}}), starts: make(chan context.Context), release: make(chan error), prefix: strings.Repeat("Read this earlier line without losing your place.\n", 80)}
	handler := newTestHandlerForRepository(t, store, generator)
	defer handler.Close()
	server := httptest.NewServer(handler)
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeoutCause(t.Context(), 45*time.Second, errors.New("mobile interaction did not settle"))
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
	settle(`!!document.querySelector('.composer textarea')`)
	check(`(() => {
   window.mobileVisible=node=>{const r=node.getBoundingClientRect();return r.width>0&&r.height>0&&r.left>=0&&r.right<=innerWidth&&r.top>=0&&r.bottom<=innerHeight;};
   const files=new DataTransfer();for(let i=0;i<12;i++)files.items.add(new File(['File text'], 'mobile-'+i+'.txt',{type:'text/plain'}));
   document.querySelector('.composer textarea').dispatchEvent(new ClipboardEvent('paste',{clipboardData:files,bubbles:true,cancelable:true}));return true;
 })()`)
	settle(`document.querySelectorAll('.attachment-strip > span').length===12 && !document.querySelector('.send-button').disabled`)
	for _, viewport := range []webuilane.Viewport{{Name: "phone", Width: 390, Height: 844}, {Name: "keyboard", Width: 390, Height: 320}, {Name: "landscape", Width: 844, Height: 390}, {Name: "small", Width: 320, Height: 568}} {
		if err := browser.SetViewport(ctx, viewport.Width, viewport.Height); err != nil {
			t.Fatal(err)
		}
		settle(`Math.abs(document.querySelector('.shell').getBoundingClientRect().height-innerHeight)<2`)
		check(`mobileVisible(document.querySelector('.composer textarea')) && mobileVisible(document.querySelector('.send-button')) && document.querySelector('.chat-log').clientHeight>0 && document.documentElement.scrollWidth<=innerWidth && document.querySelector('.composer-extras').scrollHeight>document.querySelector('.composer-extras').clientHeight`)
		check(`(() => {document.querySelector('#navigation-toggle').click();const nav=document.querySelector('#navigation'),controls=[...nav.querySelectorAll('button:not(:disabled),a[href],input')].filter(node=>node.getClientRects().length);controls.at(-1).focus();return nav.getAttribute('aria-modal')==='true'&&document.querySelector('.content').inert;})()`)
		pressKey(t, ctx, browser, "Tab", 9)
		check(`document.activeElement.id==='navigation-close'`)
		pressKey(t, ctx, browser, "Escape", 27)
		check(`document.activeElement.id==='navigation-toggle' && !document.querySelector('.content').inert`)
		check(`(() => {document.querySelector('#settings-toggle').click();return true;})()`)
		pressKey(t, ctx, browser, "Escape", 27)
		settle(`!document.querySelector('#settings-dialog').open && document.activeElement.id==='settings-toggle'`)
		if dir := os.Getenv("OVERGO_WEBUI_LANE_SCREENS"); dir != "" {
			if _, err := webuilane.CaptureState(ctx, browser, dir, viewport, "mobile-attachments"); err != nil {
				t.Fatal(err)
			}
		}
	}
	if err := browser.SetViewport(ctx, 390, 844); err != nil {
		t.Fatal(err)
	}
	settle(`Math.abs(document.querySelector('.shell').getBoundingClientRect().height-innerHeight)<2`)
	// Test safe-area geometry through Chromium's declared emulation surface.
	if err := browser.Call(ctx, "Emulation.setSafeAreaInsetsOverride", map[string]any{"insets": map[string]int{"top": 32, "left": 24, "right": 24, "bottom": 20}}, nil); err != nil {
		t.Fatal(err)
	}
	settle(`parseFloat(getComputedStyle(document.querySelector('.topbar')).paddingTop)>=32 && parseFloat(getComputedStyle(document.querySelector('.composer')).paddingBottom)>=20`)
	check(`document.querySelector('.composer textarea').getBoundingClientRect().left>=24 && document.querySelector('.send-button').getBoundingClientRect().right<=innerWidth-24`)
	if err := browser.Call(ctx, "Emulation.setSafeAreaInsetsOverride", map[string]any{"insets": map[string]int{"top": 0, "left": 0, "right": 0, "bottom": 0}}, nil); err != nil {
		t.Fatal(err)
	}
	check(`(() => {window.mobileHeight=document.documentElement.style.getPropertyValue('--viewport-height');return true;})()`)
	if err := browser.Call(ctx, "Emulation.setPageScaleFactor", map[string]any{"pageScaleFactor": 2}, nil); err != nil {
		t.Fatal(err)
	}
	settle(`visualViewport.scale===2`)
	check(`document.documentElement.style.getPropertyValue('--viewport-height')===mobileHeight && !document.querySelector('meta[name=viewport]').content.includes('user-scalable=no')`)
	if err := browser.Call(ctx, "Emulation.setPageScaleFactor", map[string]any{"pageScaleFactor": 1}, nil); err != nil {
		t.Fatal(err)
	}
	settle(`visualViewport.scale===1`)
	check(`(() => {for(const remove of document.querySelectorAll('.attachment-strip button'))remove.click();return !document.querySelector('.send-button').disabled;})()`)
	say(t, ctx, browser, "Stream a long response")
	select {
	case <-generator.starts:
	case <-ctx.Done():
		t.Fatal("response did not start")
	}
	settle(`document.querySelector('.chat-log').scrollHeight>document.querySelector('.chat-log').clientHeight && document.querySelector('.sr-only[role=status]').textContent==='Receiving a response.'`)
	check(`(() => {const log=document.querySelector('.chat-log');window.mobileStreamNode=log.querySelector('.msg.assistant .body').firstChild;log.scrollTop=0;return log.getAttribute('aria-live')==='off';})()`)
	settle(`!document.querySelector('.jump-latest').hidden`)
	check(`mobileVisible(document.querySelector('.jump-latest')) && mobileVisible(document.querySelector('.stop-button'))`)
	// Completion changes Markdown layout; the earlier reading position remains.
	select {
	case generator.release <- nil:
	case <-ctx.Done():
		t.Fatal("response did not release")
	}
	settle(`!document.querySelector('.send-button').disabled && document.querySelector('.sr-only[role=status]').textContent==='Response ready.'`)
	check(`document.querySelector('.chat-log').scrollTop===0 && !document.querySelector('.jump-latest').hidden`)
	check(`(() => {document.querySelector('.jump-latest').focus();return true;})()`)
	pressKey(t, ctx, browser, "Enter", 13)
	settle(`document.querySelector('.jump-latest').hidden && Math.ceil(document.querySelector('.chat-log').scrollTop+document.querySelector('.chat-log').clientHeight)>=document.querySelector('.chat-log').scrollHeight`)
	if findings, err := browser.LayoutAudit(ctx); err != nil || len(findings) != 0 {
		t.Fatalf("completed mobile response layout: %v, %v", findings, err)
	}
	// Exercise the shared renderer's streaming node, safe Markdown tables and keyboard disclosure.
	check(`(() => {
   const host=document.querySelector('#panel-chat');for(const node of host.querySelectorAll(':scope > .chat-log,:scope > .jump-latest,:scope > .sr-only'))node.remove();
   const thread=overgo.thread(host);window.mobileThread=thread;host.prepend(thread.node);host.insertBefore(host.querySelector('.jump-latest'),host.querySelector('.composer'));
   const message=thread.add('assistant','One');thread.renderMessage(message,true);const text=message.node.querySelector('.body').firstChild;message.content+=' two';thread.renderMessage(message,true);
   if(text!==message.node.querySelector('.body').firstChild || text.data!=='One two')return false;
   thread.renderMessage(message,false);
   const fence=String.fromCharCode(96).repeat(3);
   thread.add('assistant',fence+'text\n'+Array(80).fill('wide-code').join(' ')+'\n'+fence+'\n\n| Name | Value |\n| :--- | ---: |\n| a\\|b | <script>alert(1)</script> |');
   thread.add('assistant','|'+Array(8).fill('Column').join('|')+'|\n|'+Array(8).fill('---').join('|')+'|\n|'+Array(8).fill('Wide value').join('|')+'|');
   thread.toolCard({name:'Read document',arguments:{path:'notes.txt'}});
   const button=host.querySelector('.tool-header');button.focus();
   return !!host.querySelector('.md-table-scroll table')&&host.querySelector('td').textContent==='a|b'&&!host.querySelector('script')&&host.querySelector('.md-pre').scrollWidth>host.querySelector('.md-pre').clientWidth&&[...host.querySelectorAll('.md-table-scroll')].some(table=>table.scrollWidth>table.clientWidth);
 })()`)
	pressKey(t, ctx, browser, "Enter", 13)
	check(`document.querySelector('.tool-header').getAttribute('aria-expanded')==='true' && !document.querySelector('.tool-body').hidden && document.documentElement.scrollWidth<=innerWidth`)
	check(`(() => {mobileThread.errorRow('A recoverable error');return document.querySelector('.msg.error').getAttribute('role')==='alert';})()`)
	if dir := os.Getenv("OVERGO_WEBUI_LANE_SCREENS"); dir != "" {
		for _, scheme := range webuilane.ColourSchemes {
			if err := browser.SetColorScheme(ctx, scheme); err != nil {
				t.Fatal(err)
			}
			if _, err := webuilane.CaptureState(ctx, browser, dir, webuilane.Viewport{Name: "phone", Width: 390, Height: 844}, "mobile-reply-"+scheme); err != nil {
				t.Fatal(err)
			}
		}
	}
	// The real response was allowed to complete; the separate control test proves Stop cancels it.
	t.Log("mobile interaction leg: phone/landscape/reduced heights, paste, safe-area override, pinch zoom, reading position, Jump to latest, accessible disclosure and status passed; physical keyboards/Safari/screen readers not exercised")
}
