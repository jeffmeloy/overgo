package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"maps"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"unicode/utf8"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/inference"
	"overgo/internal/overgodb"
	"overgo/internal/tokenizer"
)

// The pair the fixture's model drafts for every passage.
const (
	draftQuestion = "What is wide?"
	draftAnswer   = "The sea is wide, and the tide comes in twice a day."
)

// draftingGenerator is the fake generator with a count of its generations.
// Its first pairless generations answer text that holds no pair, so a test
// can say which passages of a draft the model failed, and the generation
// whose count is failing ends in an error, as a device that falls over
// partway through a draft does. A failing of zero fails none.
type draftingGenerator struct {
	*fakeGenerator
	generations atomic.Int32
	pairless    int32
	failing     int32
}

// Generate counts the generation. The failing one ends in an error. While
// the count is within the generator's pairless it answers the four ways a
// model fails a passage in turn: prose with no object in it, an object cut
// off before it closes (as a reply that ran to the ceiling is), a question
// with no answer, then a whole object that is no pair.
func (generator *draftingGenerator) Generate(ctx context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	count := generator.generations.Add(1)
	if count == generator.failing {
		return nil, "", errors.New("the device fell over")
	}
	if count > generator.pairless {
		return generator.fakeGenerator.Generate(ctx, prompt, options)
	}
	return (&fakeGenerator{pieces: draftFailures[int(count)%len(draftFailures)]}).Generate(ctx, prompt, options)
}

// draftFailures are the replies that hold no pair, each as the pieces the
// model writes it in; the second is the one a first pairless generation gives.
var draftFailures = [][]string{
	{`{"note":`, `"no pair here"}`},
	{"no pair ", "here"},
	{`Here it is: {"question": `, `"cut off`},
	{`{"question":`, `"` + draftQuestion + `"}`},
}

// draftReceipt is the drafting route's answer as a client reads it.
type draftReceipt struct {
	Name     string `json:"name"`
	Template string `json:"template"`
	Model    string `json:"model"`
	Dataset  string `json:"dataset"`
	Commit   string `json:"commit"`
	Files    uint64 `json:"files"`
	Bytes    uint64 `json:"bytes"`
	Skipped  int    `json:"skipped"`
}

// draftFixture is a handler over a fresh store and a datasets root that does
// not exist yet. Its model answers one pair with prose around it, as a model
// that explains itself does, after pairless generations that hold no pair.
//
// The handler keeps the serving policy's own ceiling on a request's output,
// as a server started without a max-tokens flag does, and the reply is one
// piece a word: more pieces than the policy's default output budget, so a
// draft held to that default loses the pair's closing brace. The fixture
// fails a test whose reply stopped being longer than the default.
func draftFixture(t *testing.T, pairless int32) (*Handler, *draftingGenerator, string) {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	datasets := filepath.Join(t.TempDir(), "datasets")
	generator := &draftingGenerator{pairless: pairless, fakeGenerator: &fakeGenerator{
		pieces: strings.SplitAfter(`Here it is: {"question": "`+draftQuestion+`", "answer": "`+draftAnswer+`"} Done.`, " "),
	}}
	handler := newTestHandlerForRepository(t, store, generator, func(config *Config) {
		config.MaxTokens, config.LibraryIntake.DatasetRoot = 0, datasets
	})
	if pieces := len(generator.pieces); pieces <= handler.defaultOutputTokens || pieces > handler.config.MaxTokens {
		t.Fatalf("the fixture's reply is %d pieces, which must be more than the default output budget of %d and within the ceiling of %d",
			pieces, handler.defaultOutputTokens, handler.config.MaxTokens)
	}
	return handler, generator, datasets
}

// workspacesTold subscribes to a handler's events and answers a function
// that drains them and names the inventories every open workspace was told
// to read again since the function was last called.
func workspacesTold(t *testing.T, handler *Handler) func() []string {
	t.Helper()
	events, unsubscribe, err := handler.events.subscribe(handler.config.MaxStoredResponses)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(unsubscribe)
	return func() []string {
		inventories := []string{}
		for {
			select {
			case event := <-events:
				if change, told := event.value.(workspaceChange); told {
					inventories = append(inventories, change.Inventory)
				}
			default:
				return inventories
			}
		}
	}
}

// draftRequest posts fields to the drafting route.
func draftRequest(t *testing.T, handler http.Handler, fields map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	return serveTestRequest(handler, http.MethodPost, "/datasets/draft", marshalTestJSON(t, fields))
}

// directoryNames lists a directory's entries; a directory that was never
// made holds none.
func directoryNames(t *testing.T, directory string) []string {
	t.Helper()
	entries, err := os.ReadDir(directory)
	if err != nil && !os.IsNotExist(err) {
		t.Fatal(err)
	}
	names := []string{}
	for _, entry := range entries {
		names = append(names, entry.Name())
	}
	return names
}

// draftedPassages reads a drafted dataset's conversation files in name order
// and answers the passage each user turn carries ahead of the question. It
// fails the test when a file is anything but a user turn that ends in the
// question and an assistant turn that is the answer.
func draftedPassages(t *testing.T, directory, question, answer string) []string {
	t.Helper()
	var passages []string
	for _, name := range directoryNames(t, directory) {
		data, err := os.ReadFile(filepath.Join(directory, name))
		if err != nil {
			t.Fatal(err)
		}
		var conversation dataset.Conversation
		if err := json.Unmarshal(data, &conversation); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		messages := conversation.Messages
		if len(messages) != 2 || messages[0].Role != "user" || messages[1].Role != "assistant" || messages[1].Content != answer {
			t.Fatalf("%s holds %+v", name, messages)
		}
		passage, asked := strings.CutSuffix(messages[0].Content, "\n\n"+question)
		if !asked || filepath.Ext(name) != ".json" {
			t.Fatalf("%s asks %q", name, messages[0].Content)
		}
		passages = append(passages, passage)
	}
	return passages
}

// TestDatasetDraftingRegistersPairs holds the drafting route: the served
// model sees a template's fixed instruction and one passage of the text at a
// time, and the pair it answers for each passage becomes one conversation of
// a new dataset the catalog lists and the preview reads, with every open
// workspace told the datasets list changed. The text may be pasted or be a
// document the intake stored, a PDF among them. A request that cannot be
// drafted is refused before the model generates, leaves no directory and
// tells no workspace. An answer that holds no pair is skipped and counted,
// and a draft whose every answer is skipped registers nothing, whichever way
// each fails: prose, an object cut off, a question alone, or an object that
// is no pair. Each answer
// has the server's ceiling to fit in: the fixture's pair is longer than the
// default output budget, which would cut it short. A generation that fails
// partway ends the draft in its error with nothing registered and no
// workspace told, and leaves the server able to draft again.
func TestDatasetDraftingRegistersPairs(t *testing.T) {
	t.Parallel()
	handler, generator, datasets := draftFixture(t, 0)
	saw := func() []inference.ChatMessage {
		generator.mu.Lock()
		defer generator.mu.Unlock()
		return generator.chatMessages
	}
	changed := workspacesTold(t, handler)
	// Longer than one passage and shorter than two, with a letter that is
	// more than one byte so a cut inside it would show.
	text := strings.TrimSpace(strings.Repeat("The sea is wide, and the tide comes in twice a day at Sète. ", 50))
	first := draftRequest(t, handler, map[string]string{"name": " sea-notes ", "template": "grounded-qa", "text": text})
	var receipt draftReceipt
	if first.Code != http.StatusOK || json.Unmarshal(first.Body.Bytes(), &receipt) != nil {
		t.Fatalf("draft status=%d body=%s", first.Code, first.Body.String())
	}
	if receipt.Name != "sea-notes" || receipt.Template != "grounded-qa" || receipt.Model != testModelID || receipt.Dataset == "" ||
		len(receipt.Commit) != len(artifact.CommitID{}.String()) || receipt.Files != 2 || receipt.Bytes == 0 || receipt.Skipped != 0 || generator.generations.Load() != 2 {
		t.Fatalf("receipt = %+v after %d generations", receipt, generator.generations.Load())
	}
	if told := changed(); !slices.Equal(told, []string{"/datasets"}) {
		t.Fatalf("the draft told the open workspaces that %v changed, want the datasets list alone", told)
	}
	passages := draftedPassages(t, filepath.Join(datasets, "sea-notes"), draftQuestion, draftAnswer)
	if len(passages) != 2 || passages[0]+passages[1] != text || !utf8.ValidString(passages[0]) || len(passages[0]) < len(passages[1]) {
		t.Fatalf("the text was cut into %q", passages)
	}
	grounded := saw()
	if len(grounded) != 2 || grounded[0].Role != inference.ChatRoleSystem || !strings.Contains(grounded[0].Content, "question") || !strings.Contains(grounded[0].Content, "JSON") ||
		grounded[1].Role != inference.ChatRoleUser || grounded[1].Content != passages[1] {
		t.Fatalf("the served model saw %+v", grounded)
	}
	var catalog browseDatasetsResponse
	if listed := serveTestRequest(handler, http.MethodGet, "/datasets", ""); listed.Code != http.StatusOK || json.Unmarshal(listed.Body.Bytes(), &catalog) != nil {
		t.Fatalf("catalog status=%d body=%s", listed.Code, listed.Body.String())
	}
	if index := slices.IndexFunc(catalog.Datasets, func(entry datasetEntry) bool { return entry.Name == "sea-notes" }); index < 0 ||
		catalog.Datasets[index].Dataset != receipt.Dataset || catalog.Datasets[index].Files != 2 || !slices.Contains(catalog.Datasets[index].Formats, "json") {
		t.Fatalf("the catalog lists %+v, the receipt names %s", catalog.Datasets, receipt.Dataset)
	}
	if preview := serveTestRequest(handler, http.MethodPost, "/datasets/preview", `{"name":"sea-notes","position":0,"limit":1}`); preview.Code != http.StatusOK ||
		!strings.Contains(preview.Body.String(), draftQuestion) {
		t.Fatalf("preview status=%d body=%s", preview.Code, preview.Body.String())
	}

	// The template chooses the instruction the model drafts under.
	table := "city,degrees\nOslo,4\nSète,19"
	if drafted := draftRequest(t, handler, map[string]string{"name": "sea-sql", "template": "text-to-sql", "text": table}); drafted.Code != http.StatusOK {
		t.Fatalf("text to SQL draft status=%d body=%s", drafted.Code, drafted.Body.String())
	}
	if messages := saw(); len(messages) != 2 || messages[0].Content == grounded[0].Content || !strings.Contains(messages[0].Content, "SQL") || messages[1].Content != table {
		t.Fatalf("under the SQL template the served model saw %+v", messages)
	}

	// A stored document is drafted from as its text; a PDF as the text it draws.
	intake := func(mediaType, body string) string {
		t.Helper()
		var stored attachmentIntake
		if response := intakeRequest(handler, mediaType, body); response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &stored) != nil {
			t.Fatalf("intake of %s status=%d body=%s", mediaType, response.Code, response.Body.String())
		}
		return stored.ID
	}
	for _, document := range []struct{ name, mediaType, body, text string }{
		{"harbour", "text/plain", "The harbour is deep.", "The harbour is deep."},
		{"tide", "application/pdf", string(testPDF(t, `BT /F1 12 Tf 72 700 Td (The tide turns twice a day.) Tj ET`)), "The tide turns twice a day."},
	} {
		drafted := draftRequest(t, handler, map[string]string{"name": document.name, "template": "grounded-qa", "document": intake(document.mediaType, document.body)})
		if messages := saw(); drafted.Code != http.StatusOK || len(messages) != 2 || messages[1].Content != document.text {
			t.Fatalf("%s draft status=%d body=%s; the served model saw %+v", document.name, drafted.Code, drafted.Body.String(), messages)
		}
	}

	// Every refusal comes before the model generates and leaves no directory.
	png, err := base64.StdEncoding.DecodeString(tinyPNGBase64(t))
	if err != nil {
		t.Fatal(err)
	}
	unstored, err := artifact.IdentifyBytes(artifact.KindFile, []byte("never stored"))
	if err != nil {
		t.Fatal(err)
	}
	refusals := []struct {
		why    string
		fields map[string]string
		says   string
	}{
		{"an unknown template", map[string]string{"name": "refused", "template": "haiku", "text": "The sea is wide."}, "grounded-qa, text-to-sql"},
		{"no source", map[string]string{"name": "refused", "template": "grounded-qa"}, "exactly one of text or document"},
		{"two sources", map[string]string{"name": "refused", "template": "grounded-qa", "text": "The sea is wide.", "document": intake("text/plain", "Two sources.")}, "exactly one of text or document"},
		{"blank text", map[string]string{"name": "refused", "template": "grounded-qa", "text": " \n "}, "passages"},
		{"more text than one request drafts", map[string]string{"name": "refused", "template": "grounded-qa", "text": strings.Repeat("x", 600_000)}, "passages"},
		{"a name that is a path", map[string]string{"name": "refused/inside", "template": "grounded-qa", "text": "The sea is wide."}, "one path element"},
		{"a name that is taken", map[string]string{"name": "sea-notes", "template": "grounded-qa", "text": "Another note."}, "already exists"},
		{"an image", map[string]string{"name": "refused", "template": "grounded-qa", "document": intake("image/png", string(png))}, "not textual"},
		{"a text document that is not text", map[string]string{"name": "refused", "template": "grounded-qa", "document": intake("text/plain", "\xff\xfe no text")}, "UTF-8"},
		{"a text document that holds a NUL byte", map[string]string{"name": "refused", "template": "grounded-qa", "document": intake("text/plain", "no\x00text")}, "NUL"},
		{"a document never stored", map[string]string{"name": "refused", "template": "grounded-qa", "document": unstored.String()}, "not stored"},
		{"a document id that is none", map[string]string{"name": "refused", "template": "grounded-qa", "document": "nonsense"}, "nonsense"},
	}
	generations, _ := generator.generations.Load(), changed()
	for _, refusal := range refusals {
		if response := draftRequest(t, handler, refusal.fields); response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), refusal.says) {
			t.Errorf("%s answered %d, want %d naming %q: %s", refusal.why, response.Code, http.StatusBadRequest, refusal.says, response.Body.String())
		}
	}
	if after, told := generator.generations.Load(), changed(); after != generations || len(told) != 0 {
		t.Errorf("refused drafts reached the model %d times and told the open workspaces that %v changed", after-generations, told)
	}

	// A server that cannot draft says what it lacks. The one with no chat
	// model shares the datasets root, so the listing below covers it too.
	chatless, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = chatless.Close() })
	silent, err := New(Config{Repository: chatless, RuntimePolicy: testRuntimePolicy(), LibraryIntake: LibraryIntake{DatasetRoot: datasets}}, GenerationRefused{Reason: "No chat model is loaded"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = silent.Close() })
	storeless := newTestHandler(t, &fakeGenerator{})
	t.Cleanup(func() { _ = storeless.Close() })
	for _, unable := range []struct {
		why     string
		handler http.Handler
		says    string
	}{
		{"a server whose model cannot chat", silent, "No model that can chat is being served."},
		{"a server without a store", storeless, "library writes need the store"},
		{"a server without a datasets directory", newTestHandlerWithRepository(t, &fakeGenerator{}), "datasets directory"},
	} {
		response := draftRequest(t, unable.handler, map[string]string{"name": "unable", "template": "grounded-qa", "text": "The sea is wide."})
		if response.Code != http.StatusNotImplemented || !strings.Contains(response.Body.String(), unable.says) {
			t.Errorf("%s answered %d, want %d naming %q: %s", unable.why, response.Code, http.StatusNotImplemented, unable.says, response.Body.String())
		}
	}
	if names := directoryNames(t, datasets); !slices.Equal(names, []string{"harbour", "sea-notes", "sea-sql", "tide"}) {
		t.Errorf("after the refusals the datasets root holds %v, want the four drafted datasets alone", names)
	}

	// The first passage's answer holds no pair: it is skipped and counted,
	// and the pair that was drafted keeps the passage it was drafted from.
	skipping, _, skipped := draftFixture(t, 1)
	second := draftRequest(t, skipping, map[string]string{"name": "sea-notes", "template": "grounded-qa", "text": text})
	var partial draftReceipt
	if second.Code != http.StatusOK || json.Unmarshal(second.Body.Bytes(), &partial) != nil || partial.Files != 1 || partial.Skipped != 1 {
		t.Fatalf("a draft with one answer skipped: status=%d body=%s", second.Code, second.Body.String())
	}
	if kept := draftedPassages(t, filepath.Join(skipped, "sea-notes"), draftQuestion, draftAnswer); len(kept) != 1 || kept[0] != passages[1] {
		t.Fatalf("the pair kept the passage %q, want the second passage", kept)
	}

	// No passage's answer holds a pair, and the text is long enough for the
	// model to fail in each of its four ways: nothing is registered.
	failing, failed, empty := draftFixture(t, draftMaxPassages)
	none := draftRequest(t, failing, map[string]string{"name": "sea-notes", "template": "grounded-qa", "text": strings.Repeat(text+" ", len(passages)+1)})
	if none.Code != http.StatusInternalServerError || !strings.Contains(none.Body.String(), "no question and answer pair") || len(directoryNames(t, empty)) != 0 {
		t.Fatalf("a draft with no pair: status=%d body=%s root=%v", none.Code, none.Body.String(), directoryNames(t, empty))
	}
	if ways := int(failed.generations.Load()); ways < len(draftFailures) {
		t.Fatalf("the draft with no pair made %d generations, too few to fail in each of the %d ways", ways, len(draftFailures))
	}

	// The second passage's generation fails after the first drafted a pair:
	// the draft ends in the failure, registers nothing and tells no
	// workspace. The session is free, looked at without waiting so a draft
	// that kept it is named here and not by a request that never returns,
	// and the name is not taken, so the same request drafts when it is sent
	// again.
	broken, device, bare := draftFixture(t, 0)
	device.failing = 2
	told := workspacesTold(t, broken)
	fell := draftRequest(t, broken, map[string]string{"name": "sea-notes", "template": "grounded-qa", "text": text})
	if left, heard := directoryNames(t, bare), told(); fell.Code != http.StatusInternalServerError || !strings.Contains(fell.Body.String(), "the device fell over") ||
		device.generations.Load() != device.failing || len(left) != 0 || len(heard) != 0 {
		t.Fatalf("a draft whose second generation failed: status=%d body=%s after %d generations, root=%v, workspaces told %v",
			fell.Code, fell.Body.String(), device.generations.Load(), left, heard)
	}
	session, free := broken.acquireSession(-1)
	if !free {
		t.Fatal("the draft whose generation failed kept the session")
	}
	broken.releaseSession(session)
	again := draftRequest(t, broken, map[string]string{"name": "sea-notes", "template": "grounded-qa", "text": text})
	if files, heard := directoryNames(t, filepath.Join(bare, "sea-notes")), told(); again.Code != http.StatusOK || len(files) != 2 || !slices.Equal(heard, []string{"/datasets"}) {
		t.Fatalf("the draft after a failed one: status=%d body=%s files=%v, workspaces told %v", again.Code, again.Body.String(), files, heard)
	}
}

// TestDatasetDraftingLeavesNoWorkingFile holds the route to cleaning up after
// itself: with the system's temporary directory pointed at an empty one, a
// draft that registers leaves that directory empty. The test changes the
// environment, so it does not run beside the others.
func TestDatasetDraftingLeavesNoWorkingFile(t *testing.T) {
	working := t.TempDir()
	for _, variable := range []string{"TMP", "TEMP", "TMPDIR"} {
		t.Setenv(variable, working)
	}
	handler, _, datasets := draftFixture(t, 0)
	drafted := draftRequest(t, handler, map[string]string{"name": "sea-notes", "template": "grounded-qa", "text": "The sea is wide."})
	if drafted.Code != http.StatusOK || len(directoryNames(t, filepath.Join(datasets, "sea-notes"))) != 1 {
		t.Fatalf("draft status=%d body=%s", drafted.Code, drafted.Body.String())
	}
	if left := directoryNames(t, working); len(left) != 0 {
		t.Fatalf("the draft left %v in the temporary directory", left)
	}
}

// TestDatasetDraftingPageOffersTheAction holds the Datasets tab to the
// drafting route. The page posts a draft to it, and offers each template the
// server holds and no other: a template added to the server with no choice on
// the page, or a choice the server would refuse, fails here. This reads the
// page's source; what pressing Draft dataset does is held in the browser
// (TestWebUIBrowserDatasetDrafting).
func TestDatasetDraftingPageOffersTheAction(t *testing.T) {
	t.Parallel()
	page := webuiJavaScript(t)["mod/datasets.js"]
	table := regexp.MustCompile(`const draftTemplates = \[(.*)\];`).FindStringSubmatch(page)
	if table == nil {
		t.Fatal("the Datasets tab declares no drafting templates")
	}
	var offered []string
	for _, choice := range regexp.MustCompile(`\["([^"]+)", "`).FindAllStringSubmatch(table[1], -1) {
		offered = append(offered, choice[1])
	}
	slices.Sort(offered)
	if held := slices.Sorted(maps.Keys(datasetDraftTemplates)); len(held) == 0 || !slices.Equal(held, offered) {
		t.Errorf("the server drafts under the templates %v and the Datasets tab offers %v; want the page to offer each template the server holds and no other", held, offered)
	}
	if !strings.Contains(page, `overgo.api.post("/datasets/draft", request`) || !strings.Contains(page, `overgo.api.upload("/artifacts/intake", file`) {
		t.Error("the Datasets tab does not post a draft to the drafting route, or does not store a chosen document through the artifact intake")
	}
}
