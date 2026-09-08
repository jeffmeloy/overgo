package server

import (
	"context"
	"errors"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
	"overgo/internal/webuilane"
)

func TestWebUIBrowserConversationActions(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testevidence.ShortIntegrationSkip + ": conversation actions run through cmd/webui-lane")
	}
	seed, _, _ := conversationHistoryFixture(t)
	description, ok := seed.interactionDescription()
	if !ok {
		t.Fatal("fixture interaction identity unavailable")
	}
	if _, err := runrecord.PublishInteraction(t.Context(), seed.repository, runrecord.Interaction{Response: "resp_media", Recipe: description.Identity.Recipe, Model: description.Identity.Model, Node: description.Interaction.Node},
		[]runrecord.InteractionMessage{{Role: "user", Content: "前after", Media: []runrecord.InteractionMedia{{Type: "image", Data: "https://example.invalid/image.png", TextOffset: len("前")}}}, {Role: "user", Content: "Second input"}, {Role: "assistant", Content: "Media fixture answer"}}, runrecord.OutcomeSucceeded); err != nil {
		t.Fatal(err)
	}
	if err := seed.Close(); err != nil {
		t.Fatal(err)
	}
	generator := &responseControlGenerator{recipeInspectorGenerator: responseRecipeGenerator(t, &fakeGenerator{pieces: []string{"Branched answer"}}), starts: make(chan context.Context), release: make(chan error), prefix: "Branch prefix "}
	handler := newTestHandlerForRepository(t, seed.repository, generator)
	server := httptest.NewServer(handler)
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeoutCause(t.Context(), 45*time.Second, errors.New("conversation actions did not settle"))
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
	finish := func() {
		t.Helper()
		select {
		case <-generator.starts:
		case <-ctx.Done():
			t.Fatal("branch execution did not start")
		}
		select {
		case generator.release <- nil:
		case <-ctx.Done():
			t.Fatal("branch execution did not release")
		}
	}
	if err := browser.SetViewport(ctx, 390, 844); err != nil {
		t.Fatal(err)
	}
	settle(`!!document.querySelector('.composer textarea')`)
	check(`(async () => {
  window.actionButton=text=>[...document.querySelectorAll('#panel-chat button,#conversation-settings button')].find(node=>node.textContent===text);
  window.actionSource=(await overgo.api.get('/interactions?view=active&q=Older')).conversations[0];
  window.actionOriginal=(await overgo.api.get('/interactions/messages?response='+actionSource.latest)).messages;
  window.actionStream=overgo.api.stream;window.actionRequests=[];
  overgo.api.stream=async function(path,body,options){if(path==='/v1/responses')actionRequests.push(body);return actionStream.call(this,path,body,options);};
  overgo.openConversation(actionSource);return true;
})()`)
	settle(`document.querySelectorAll('.chat-log .msg').length===10 && !document.querySelector('.send-button').disabled`)
	check(`(() => {
  const input=document.querySelector('.composer textarea');input.value='Unsent composer draft';input.dispatchEvent(new Event('input',{bubbles:true}));
  document.querySelectorAll('.msg.user')[1].querySelector('.turn-action').click();return true;
})()`)
	settle(`!!document.querySelector('.turn-editor textarea')`)
	check(`(() => {document.querySelector('.turn-editor textarea').value='Edited second turn';const button=actionButton('Send edited message');button.click();button.click();return true;})()`)
	finish()
	settle(`!document.querySelector('.send-button').disabled && document.querySelector('.chat-log').textContent.includes('Branched answer')`)
	check(`(async () => {
  const chain=await overgo.api.get('/interactions/messages?response='+overgo.conversation().latest);
  const original=await overgo.api.get('/interactions/messages?response='+actionSource.latest);
  window.editedBranch={...overgo.conversation()};
  return actionRequests.length===1&&actionRequests[0].previous_response_id===actionOriginal[0].response&&chain.messages.length===4&&chain.messages[2].content==='Edited second turn'&&original.messages.length===10&&original.messages[2].content==='Next history turn'&&document.querySelector('.composer textarea').value==='Unsent composer draft';
})()`)
	check(`(() => {actionButton('Return to original').click();return true;})()`)
	settle(`document.querySelectorAll('.chat-log .msg').length===10`)
	check(`(() => {document.querySelectorAll('.msg.assistant .turn-action')[4].click();return true;})()`)
	finish()
	settle(`!document.querySelector('.send-button').disabled && document.querySelector('.chat-log').textContent.includes('Branched answer')`)
	check(`(async () => {const chain=await overgo.api.get('/interactions/messages?response='+overgo.conversation().latest);return actionRequests.length===2&&actionRequests[1].previous_response_id===actionOriginal[6].response&&chain.messages.length===10&&chain.messages[8].content===actionOriginal[8].content;})()`)
	check(`(() => {actionButton('Return to original').click();return true;})()`)
	settle(`document.querySelectorAll('.chat-log .msg').length===10 && !document.querySelector('.send-button').disabled`)
	check(`(() => {
  let fail=true;overgo.api.stream=async function(path,body,options){if(path==='/v1/responses'&&fail){fail=false;throw Object.assign(new Error('Branch temporarily unavailable'),{status:503});}if(path==='/v1/responses')actionRequests.push(body);return actionStream.call(this,path,body,options);};
  document.querySelectorAll('.msg.assistant .turn-action')[4].click();return true;
})()`)
	settle(`!!actionButton('Retry branch') && !actionButton('Retry branch').disabled`)
	check(`document.querySelectorAll('.chat-log .msg.assistant').length===5 && document.querySelector('.chat-log').textContent.includes('Original conversation restored') && overgo.conversation().latest===actionSource.latest && document.querySelector('.composer textarea').value==='Unsent composer draft'`)
	check(`(() => {actionButton('Retry branch').click();return true;})()`)
	finish()
	settle(`!document.querySelector('.send-button').disabled && document.querySelector('.chat-log').textContent.includes('Branched answer')`)
	// Stop cancels the actual new branch, with the original still readable.
	check(`(() => {actionButton('Return to original').click();return true;})()`)
	settle(`document.querySelectorAll('.chat-log .msg').length===10 && !document.querySelector('.send-button').disabled`)
	check(`(() => {document.querySelectorAll('.msg.assistant .turn-action')[4].click();return true;})()`)
	var execution context.Context
	select {
	case execution = <-generator.starts:
	case <-ctx.Done():
		t.Fatal("stoppable branch did not start")
	}
	settle(`!document.querySelector('.stop-button').hidden`)
	check(`document.activeElement===document.querySelector('.stop-button')`)
	pressKey(t, ctx, browser, "Enter", 13)
	select {
	case <-execution.Done():
	case <-ctx.Done():
		t.Fatal("Stop did not cancel branch execution")
	}
	settle(`!document.querySelector('.send-button').disabled && document.querySelector('.chat-log').textContent.includes('Stopped')`)
	check(`(() => {overgo.openConversation(editedBranch);return true;})()`)
	settle(`document.querySelectorAll('.chat-log .msg').length===4 && document.querySelector('.chat-log').textContent.includes('Edited second turn')`)
	check(`(() => {
  document.querySelector('#settings-toggle').click();[...document.querySelectorAll('#conversation-settings summary')].find(node=>node.textContent==='Copy or export conversation').click();
  Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async text=>{throw new Error('Clipboard denied');}}});actionButton('Copy conversation').click();return true;
})()`)
	settle(`document.querySelector('#conversation-settings [role=alert]').textContent.includes('Clipboard denied')`)
	check(`(() => {navigator.clipboard.writeText=async text=>{window.actionCopy=text;};actionButton('Copy conversation').click();return true;})()`)
	settle(`typeof actionCopy==='string' && actionCopy.includes('Edited second turn')`)
	check(`actionCopy.includes('## Assistant') && !actionCopy.includes('Unsent composer draft') && !actionCopy.includes('Next history turn')`)
	check(`(() => {
  window.actionCreateURL=URL.createObjectURL;window.actionLinkClick=HTMLAnchorElement.prototype.click;
  URL.createObjectURL=blob=>{blob.text().then(text=>window.actionExport=text);return actionCreateURL(blob);};
  HTMLAnchorElement.prototype.click=function(){if(this.download.endsWith('.md'))window.actionFilename=this.download;else actionLinkClick.call(this);};
  actionButton('Export Markdown').click();return true;
})()`)
	settle(`typeof actionExport==='string' && !!actionFilename`)
	check(`actionExport===actionCopy && actionFilename.endsWith('.md')`)
	check(`(() => {URL.createObjectURL=actionCreateURL;HTMLAnchorElement.prototype.click=actionLinkClick;document.querySelector('#settings-dialog').close();return true;})()`)
	for _, viewport := range []webuilane.Viewport{{Name: "small", Width: 320, Height: 568}, {Name: "phone", Width: 390, Height: 844}, {Name: "desktop", Width: 1280, Height: 800}} {
		if findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "conversation-actions"); err != nil || len(findings) != 0 {
			t.Fatalf("actions %s layout: %v, %v", viewport.Name, findings, err)
		}
	}
	// Editing the root creates a new root and copies the unsent draft; the
	// original conversation retains its own draft and immutable transcript.
	check(`(() => {overgo.openConversation(actionSource);return true;})()`)
	settle(`document.querySelectorAll('.chat-log .msg').length===10 && !document.querySelector('.send-button').disabled`)
	check(`(() => {document.querySelector('.msg.user .turn-action').click();return true;})()`)
	settle(`!!document.querySelector('.turn-editor textarea')`)
	pressKey(t, ctx, browser, "Escape", 27)
	check(`!document.querySelector('.turn-editor') && document.activeElement===document.querySelector('.msg.user .turn-action') && document.querySelector('.composer textarea').value==='Unsent composer draft'`)
	check(`(() => {document.querySelector('.msg.user .turn-action').click();return true;})()`)
	settle(`!!document.querySelector('.turn-editor textarea')`)
	check(`(() => {document.querySelector('.turn-editor textarea').value='Edited root';actionButton('Send edited message').click();return true;})()`)
	finish()
	settle(`!document.querySelector('.send-button').disabled && document.querySelector('.chat-log').textContent.includes('Branched answer')`)
	check(`overgo.conversation().root===overgo.conversation().latest && overgo.conversation().root!==actionSource.root && document.querySelector('.composer textarea').value==='Unsent composer draft'`)
	check(`(() => {actionButton('Return to original').click();return true;})()`)
	settle(`document.querySelectorAll('.chat-log .msg').length===10 && document.querySelector('.composer textarea').value==='Unsent composer draft'`)
	check(`(() => {document.querySelectorAll('.msg.user .turn-action')[1].click();return true;})()`)
	settle(`!!document.querySelector('.turn-editor textarea')`)
	check(`(() => {document.querySelector('.turn-editor textarea').value='Pending edit';return true;})()`)
	say(t, ctx, browser, "Continuation while editing")
	finish()
	settle(`!document.querySelector('.send-button').disabled && document.querySelectorAll('.chat-log .msg').length===12`)
	check(`(() => {
  window.newerSelection={...overgo.conversation()};const input=document.querySelector('.composer textarea');input.value='Newer draft';input.dispatchEvent(new Event('input',{bubbles:true}));
  overgo.api.stream=async()=>{throw Object.assign(new Error('Pending edit refused'),{status:503});};actionButton('Send edited message').click();return true;
})()`)
	settle(`!!actionButton('Retry branch') && !actionButton('Retry branch').disabled`)
	check(`overgo.conversation().latest===newerSelection.latest && document.querySelector('.chat-log').textContent.includes('Continuation while editing') && document.querySelectorAll('.msg.assistant').length===6 && document.querySelector('.composer textarea').value==='Newer draft'`)
	// The stored multimodal fixture is real; its generation transport refuses
	// deliberately. No network fetch or image-model claim is made by this leg.
	check(`(async () => {
  overgo.api.stream=async function(path,body){window.mediaBranchRequest=body;throw Object.assign(new Error('Media branch refused'),{status:503});};
  window.actionPost=overgo.api.post;overgo.api.post=async function(path,body,options){if(path==='/v1/responses/input_tokens')return null;return actionPost.call(this,path,body,options);};
  const source=(await overgo.api.get('/interactions?q='+encodeURIComponent('前after'))).conversations[0];overgo.openConversation(source);return true;
})()`)
	settle(`document.querySelectorAll('.chat-log .msg').length===3 && !document.querySelector('.send-button').disabled`)
	check(`(() => {document.querySelector('.msg.assistant .turn-action').click();return true;})()`)
	settle(`!!actionButton('Retry branch') && !actionButton('Retry branch').disabled`)
	check(`mediaBranchRequest.input.length===2 && mediaBranchRequest.input[0].content[0].text==='前' && mediaBranchRequest.input[0].content[1].image_url==='https://example.invalid/image.png' && mediaBranchRequest.input[0].content[2].text==='after' && mediaBranchRequest.input[1].content[0].text==='Second input'`)
	check(`(() => {document.querySelector('.msg.user .turn-action').click();return true;})()`)
	settle(`!!document.querySelector('.turn-editor textarea')`)
	check(`(() => {document.querySelector('.turn-editor textarea').value='Edited media question';actionButton('Send edited message').click();return true;})()`)
	settle(`!!actionButton('Retry branch') && !actionButton('Retry branch').disabled`)
	check(`mediaBranchRequest.input[0].content[0].text==='Edited media question' && mediaBranchRequest.input[0].content[1].type==='input_image' && mediaBranchRequest.input[1].content[0].text==='Second input' && document.querySelectorAll('.msg.user')[0].querySelector('.body').textContent==='前after'`)
	check(`(() => {document.querySelector('#settings-toggle').click();[...document.querySelectorAll('#conversation-settings summary')].find(node=>node.textContent==='Copy or export conversation').click();actionButton('Copy conversation').click();return true;})()`)
	settle(`actionCopy.includes('Attachment: image')`)
	check(`actionCopy.includes('/interactions/inspect?response=resp_media') && !actionCopy.includes('https://example.invalid/image.png')`)
	check(`(() => {document.querySelector('#settings-dialog').close();overgo.api.post=actionPost;return true;})()`)
	check(`(async () => {const other=(await overgo.api.get('/interactions?q=Other')).conversations[0];overgo.openConversation(other);return true;})()`)
	settle(`document.querySelector('.chat-log').textContent.includes('An answer from another model')`)
	check(`[...document.querySelectorAll('.msg .turn-action')].every(button=>button.disabled) && document.querySelector('.send-button').disabled && overgo.errors.length===0`)
	t.Log("conversation actions leg: real immutable edit/regenerate branches, duplicate-submit guard, failed-branch retry, actual Stop cancellation, original/branch reopening, draft preservation, clipboard refusal and Markdown download contents passed; model output is synthetic and physical devices are outside this Chromium leg")
}
