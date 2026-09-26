package server

import (
	"context"
	"net/http/httptest"
	"os"
	"slices"
	"testing"

	"overgo/internal/inference"
	"overgo/internal/overgodb"
	"overgo/internal/testskip"
	"overgo/internal/tokenizer"
	"overgo/internal/webuilane"
)

// adapterGenerator: a served model with one loaded adapter; a run with the
// adapter enabled answers "adapted reply", one with it disabled "base reply".
type adapterGenerator struct {
	*recipeInspectorGenerator
}

func (adapterGenerator) LoRAAdapters() []inference.LoRAAdapterInfo {
	return []inference.LoRAAdapterInfo{{ID: 0, Path: "adapters/tone.gguf", Scale: 1}}
}

func (adapterGenerator) SetLoRAScales(context.Context, []inference.LoRAScale) error { return nil }

func (adapterGenerator) Generate(_ context.Context, _ string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	words := []string{"adapted ", "reply"}
	if options.LoRAConfigured && !slices.ContainsFunc(options.LoRA, func(scale inference.LoRAScale) bool { return scale.Scale != 0 }) {
		words[0] = "base "
	}
	ids := slices.Clone(options.PromptTokenIDs)
	for index, word := range words {
		ids = append(ids, tokenizer.TokenID(100+index))
		if err := options.OnToken(inference.TokenEvent{ID: tokenizer.TokenID(100 + index), Piece: word, Index: index}); err != nil {
			return ids, "", err
		}
	}
	return ids, "", nil
}

// TestWebUIBrowserAdapterCompare drives the adapter comparison in the chat:
// a turn's Compare adapter runs its prompt once on the base model and once
// on the loaded adapter, with one seed and unstored, and shows the two
// outputs side by side under the turn.
func TestWebUIBrowserAdapterCompare(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the adapter compare leg uses Chromium through cmd/webui-lane")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	handler := newTestHandlerForRepository(t, store, adapterGenerator{responseRecipeGenerator(t, &fakeGenerator{})})
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
	settle(`!!document.querySelector('.composer textarea')`)
	check(`(() => {
  window.compareButton = () => [...document.querySelectorAll('.chat-log .msg.assistant button')].find(button => button.textContent === 'Compare adapter');
  window.compareColumns = () => [...document.querySelectorAll('.adapter-compare .compare-column')].map(column => [column.querySelector('strong').textContent, column.querySelector('.compare-text').textContent]);
  window.comparePost = overgo.api.post; window.compareRequests = [];
  overgo.api.post = async function (path, body, options) { if (path === '/v1/responses') compareRequests.push(body); return comparePost.call(this, path, body, options); };
  const input = document.querySelector('.composer textarea');
  input.value = 'say hello'; input.dispatchEvent(new Event('input', { bubbles: true }));
  document.querySelector('.send-button').click();
  return true;
})()`)

	// The turn, made with the loaded scales, offers the comparison.
	settle(`[...document.querySelectorAll('.chat-log .msg.assistant .body')].some(body => body.textContent.includes('adapted reply')) && !!compareButton() && !compareButton().disabled`)
	check(`(() => { compareButton().click(); return true; })()`)

	// Base and adapter stand side by side, from one prompt, sampling and seed, unstored.
	settle(`JSON.stringify(compareColumns()) === JSON.stringify([['Base model', 'base reply'], ['tone.gguf', 'adapted reply']]) && !compareButton().disabled`)
	check(`(() => {
  const [base, adapted] = compareRequests;
  const same = (key) => JSON.stringify(base[key]) === JSON.stringify(adapted[key]);
  return compareRequests.length === 2 && same('input') && same('seed') && same('temperature') && same('previous_response_id') && Number.isInteger(base.seed) &&
    base.store === false && adapted.store === false && JSON.stringify(base.lora) === '[]' && JSON.stringify(adapted.lora) === '[{"id":0,"scale":1}]';
})()`)
	webuilane.Leg(t, "adapter compare leg", "a turn's Compare adapter ran its prompt on the base model and on the loaded adapter with one seed, unstored, and showed the two outputs side by side under the turn")
}
