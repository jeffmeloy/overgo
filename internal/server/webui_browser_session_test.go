package server

import (
	"context"
	"net/http/httptest"
	"os"
	"slices"
	"sync/atomic"
	"testing"

	"overgo/internal/inference"
	"overgo/internal/overgodb"
	"overgo/internal/testskip"
	"overgo/internal/tokenizer"
	"overgo/internal/webuilane"
)

// sessionWords: the sentence a sessionGenerator speaks, word i as token 100+i.
var sessionWords = []string{"The ", "quick ", "brown ", "fox ", "jumps ", "over ", "the ", "dog."}

// sessionGenerator: a fake that continues token-exactly. A prompt ending in
// sentence tokens resumes after them with all but its last token cached; an
// armed run blocks after three words until cancelled and returns the ids it
// decoded, as Stop leaves a real run.
type sessionGenerator struct {
	*recipeInspectorGenerator
	armed   atomic.Bool
	blocked chan struct{}
}

func (generator *sessionGenerator) SupportsPromptCache() bool { return true }

func (generator *sessionGenerator) Generate(ctx context.Context, _ string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	ids := []tokenizer.TokenID{10, 20}
	if options.PromptTokenIDs != nil {
		ids = slices.Clone(options.PromptTokenIDs)
	}
	spoken := 0
	for spoken < len(ids) && ids[len(ids)-1-spoken] >= 100 {
		spoken++
	}
	cached := 0
	if spoken > 0 {
		cached = len(ids) - 1
	}
	if options.OnPromptEvaluated != nil {
		options.OnPromptEvaluated(inference.PromptEvaluation{Tokens: len(ids), Cached: cached})
	}
	armed := generator.armed.CompareAndSwap(true, false)
	for index := spoken; index < len(sessionWords) && index-spoken < options.MaxNewTokens; index++ {
		ids = append(ids, tokenizer.TokenID(100+index))
		if err := options.OnToken(inference.TokenEvent{ID: tokenizer.TokenID(100 + index), Piece: sessionWords[index], Index: index - spoken}); err != nil {
			return ids, "", err
		}
		if armed && index == 2 {
			close(generator.blocked)
			<-ctx.Done()
			return ids, "", context.Cause(ctx)
		}
	}
	return ids, "", nil
}

// TestWebUIBrowserSessionContinuation drives Stop and Continue in the chat:
// a turn stopped after three words shows its note with Continue, continuing
// asks the server to resume the stopped turn (no new input), the words
// continue in the same assistant message, and the reopened conversation
// shows the one whole reply.
func TestWebUIBrowserSessionContinuation(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the session continuation leg uses Chromium through cmd/webui-lane")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	generator := &sessionGenerator{recipeInspectorGenerator: responseRecipeGenerator(t, &fakeGenerator{}), blocked: make(chan struct{})}
	generator.armed.Store(true)
	handler := newTestHandlerForRepository(t, store, generator)
	server := httptest.NewServer(handler)
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
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
	const helpers = `(() => {
  window.sessionReplies = () => [...document.querySelectorAll('.chat-log .msg.assistant')].map(message => [...message.querySelector('.body').childNodes].filter(node => !node.classList?.contains('cursor')).map(node => node.textContent).join('').trim());
  window.sessionContinue = () => [...document.querySelectorAll('.chat-log button')].find(button => button.textContent === 'Continue');
  window.sessionStopped = () => [...document.querySelectorAll('.chat-log .note')].some(note => note.textContent.startsWith('Stopped'));
  return true;
})()`
	settle(`!!document.querySelector('.composer textarea')`)
	check(helpers)
	check(`(() => {
  window.sessionStream = overgo.api.stream; window.sessionRequests = [];
  overgo.api.stream = async function (path, body, options) { if (path === '/v1/responses') sessionRequests.push(body); return sessionStream.call(this, path, body, options); };
  const input = document.querySelector('.composer textarea');
  input.value = 'say the sentence'; input.dispatchEvent(new Event('input', { bubbles: true }));
  document.querySelector('.send-button').click();
  return true;
})()`)

	// Stopped after three words, the turn offers to continue.
	select {
	case <-generator.blocked:
	case <-ctx.Done():
		t.Fatal("the turn did not reach its third word")
	}
	settle(`sessionReplies().at(-1) === 'The quick brown' && !document.querySelector('.stop-button').disabled`)
	check(`(() => { document.querySelector('.stop-button').click(); return true; })()`)
	settle(`sessionStopped() && !!sessionContinue() && !document.querySelector('.send-button').disabled`)

	// Continued, the words resume in the same message from the stopped turn.
	check(`(() => { sessionContinue().click(); return true; })()`)
	settle(`sessionReplies().length === 1 && sessionReplies()[0] === 'The quick brown fox jumps over the dog.' && !sessionStopped() && !sessionContinue()`)
	check(`(() => { const body = sessionRequests.at(-1); return body.continue === true && !!body.previous_response_id && body.input === undefined; })()`)

	// Reopened, the conversation holds the one whole reply.
	if err := browser.Evaluate(ctx, `location.reload()`, nil); err != nil {
		t.Fatal(err)
	}
	settle(`!!document.querySelector('.composer textarea') && document.querySelectorAll('.chat-log .msg.assistant').length === 1`)
	check(helpers)
	check(`sessionReplies()[0] === 'The quick brown fox jumps over the dog.' && !sessionStopped() && document.querySelectorAll('.chat-log .msg.user').length === 1`)
	webuilane.Leg(t, "session continuation leg", "a turn stopped after three words offered Continue, continuing resumed the stopped turn without new input in the same message, and the reopened conversation held the one whole reply")
}
