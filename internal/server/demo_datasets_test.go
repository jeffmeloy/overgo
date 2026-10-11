package server

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/dataset"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
	"overgo/internal/trainingdata"
)

// demoIdentity is the demo set that teaches a model to say who it is: an assistant
// with the made-up name Pebble, made as a demonstration in the overgo workbench.
const demoIdentity = "demo-identity"

// demoSets are the demo datasets the server carries, by the name each registers
// under: the identity set and a small instruction set.
var demoSets = []string{demoIdentity, "demo-instruction"}

const (
	// demoContext is the context length of the tiny model the demo sets train: with a
	// token a byte, every demo conversation fits one window of it.
	demoContext = 256
	// demoSteps is how many updates the tiny model takes on a demo set, as many as
	// the Train tab's own fixture takes. A run must leave the model answering the
	// held-out conversations otherwise than the base does, or there is no verdict
	// to give; on these sets one update was seen to do that already.
	demoSteps = 4
)

// demoRecord is one record of a demo file, in the alpaca form.
type demoRecord struct {
	Instruction string `json:"instruction"`
	Input       string `json:"input"`
	Output      string `json:"output"`
}

// prompt is the user's turn an import makes of the record.
func (record demoRecord) prompt() string {
	if record.Input == "" {
		return record.Instruction
	}
	return record.Instruction + "\n\n" + record.Input
}

// demoReceipt is what the register route answers for a dataset.
type demoReceipt struct {
	Name    string      `json:"name"`
	Dataset artifact.ID `json:"dataset"`
	Files   int         `json:"files"`
	Formats []string    `json:"formats"`
}

// demoSetsOnDisk names the demo files beside the package, in name order, each by
// the name it registers under: its file name without the extension.
func demoSetsOnDisk(t *testing.T) []string {
	t.Helper()
	entries, err := os.ReadDir("demo")
	if err != nil {
		t.Errorf("the demo sets cannot be listed: %v", err)
	}
	var names []string
	for _, entry := range entries {
		names = append(names, strings.TrimSuffix(entry.Name(), ".jsonl"))
	}
	return names
}

// demoRecords reads the records of one demo file on disk, which holds alpaca
// fields alone.
func demoRecords(t *testing.T, name string) []demoRecord {
	t.Helper()
	file, err := os.Open(filepath.Join("demo", name+".jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	decoder := json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	var records []demoRecord
	for decoder.More() {
		var record demoRecord
		if err := decoder.Decode(&record); err != nil {
			t.Fatalf("%s record %d: %v", name, len(records)+1, err)
		}
		records = append(records, record)
	}
	return records
}

// registerDemo asks the register route for the demo set of that name and returns
// the status, the body and the receipt the body holds.
func registerDemo(handler http.Handler, name string) (int, string, demoReceipt) {
	response := serveTestRequest(handler, http.MethodPost, "/library/register", fmt.Sprintf(`{"kind":"dataset","demo":%q}`, name))
	var receipt demoReceipt
	_ = json.Unmarshal(response.Body.Bytes(), &receipt)
	return response.Code, response.Body.String(), receipt
}

// TestDemoDatasetsRegisterInOneAction: one request naming a demo set registers
// it, with no file on disk to point at. Each demo file beside the package
// registers under its own name as conversations, one for each record, lists as an
// available dataset the Train tab's picker offers, and previews as a conversation
// holding its first record. A second request for the same set answers with the
// dataset it already is, and so does a request on a store that lists nothing while
// the set's directory is still on disk, which then lists it again; a name that is
// no demo set is refused and writes nothing; a server with no datasets directory
// says it needs one; and the shell that answers while no model serves registers a
// set the same way. An import of a file by its path, which shares the route, still
// works.
func TestDemoDatasetsRegisterInOneAction(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	datasets := t.TempDir()
	handler := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{}), func(config *Config) {
		config.LibraryIntake.DatasetRoot = datasets
	})
	if onDisk := demoSetsOnDisk(t); !slices.Equal(onDisk, demoSets) {
		t.Errorf("the demo files on disk name %v, want the sets %v", onDisk, demoSets)
	}
	for _, name := range demoSets {
		status, body, receipt := registerDemo(handler, name)
		if status != http.StatusOK {
			t.Fatalf("registering the demo set %s: status=%d body=%s", name, status, body)
		}
		records := demoRecords(t, name)
		if receipt.Name != name || receipt.Dataset.Kind() != artifact.KindDataset || receipt.Files != len(records) || len(records) == 0 || !slices.Equal(receipt.Formats, []string{"alpaca"}) {
			t.Fatalf("the receipt for %s is %s, want that name, a dataset, %d conversations and the alpaca format", name, body, len(records))
		}
		preview := serveTestRequest(handler, http.MethodPost, "/datasets/preview", fmt.Sprintf(`{"name":%q,"position":0,"limit":1}`, name))
		var page trainingdata.PreviewResult
		if err := json.Unmarshal(preview.Body.Bytes(), &page); preview.Code != http.StatusOK || err != nil || len(page.Examples) != 1 || len(page.Examples[0].Values) != 1 {
			t.Fatalf("previewing %s: status=%d body=%s", name, preview.Code, preview.Body.String())
		}
		var conversation dataset.Conversation
		want := []dataset.ChatMessage{{Role: "user", Content: records[0].prompt()}, {Role: "assistant", Content: records[0].Output}}
		if err := json.Unmarshal([]byte(page.Examples[0].Values[0].Text), &conversation); err != nil || !slices.Equal(conversation.Messages, want) {
			t.Fatalf("the preview of %s shows %q (%v), want its first record as the conversation %+v", name, page.Examples[0].Values[0].Text, err, want)
		}
		if status, body, again := registerDemo(handler, name); status != http.StatusOK || again.Dataset != receipt.Dataset || again.Files != receipt.Files {
			t.Errorf("registering %s a second time: status=%d body=%s, want the dataset %s it already is", name, status, body, receipt.Dataset)
		}
		// A store that lists nothing, over the datasets directory the demo was written to.
		restored, err := overgodb.Open(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		defer restored.Close()
		over := newTestHandlerForRepository(t, restored, responseRecipeGenerator(t, &fakeGenerator{}), func(config *Config) {
			config.LibraryIntake.DatasetRoot = datasets
		})
		status, body, left := registerDemo(over, name)
		relisted := serveTestRequest(over, http.MethodGet, "/datasets", "")
		if status != http.StatusOK || left.Dataset != receipt.Dataset || !strings.Contains(relisted.Body.String(), `"`+name+`"`) {
			t.Errorf("registering %s on a store that lost it, its directory still on disk: status=%d body=%s listing=%s, want the dataset %s listed again", name, status, body, relisted.Body.String(), receipt.Dataset)
		}
	}

	listing := serveTestRequest(handler, http.MethodGet, "/datasets", "")
	var catalog browseDatasetsResponse
	if err := json.Unmarshal(listing.Body.Bytes(), &catalog); listing.Code != http.StatusOK || err != nil || catalog.Count != len(demoSets) {
		t.Fatalf("the dataset listing: status=%d body=%s, want the %d demo sets", listing.Code, listing.Body.String(), len(demoSets))
	}
	for index, entry := range catalog.Datasets {
		if entry.Name != demoSets[index] || !entry.Available || !slices.Equal(entry.Formats, []string{sftDatasetFormat}) {
			t.Errorf("listed dataset %d is %+v, want %s available in the format the Train tab's picker offers", index, entry, demoSets[index])
		}
	}

	if status, body, _ := registerDemo(handler, "demo-absent"); status != http.StatusUnprocessableEntity || !strings.Contains(body, "demo-absent") {
		t.Errorf("a name that is no demo set: status=%d body=%s, want it refused by name", status, body)
	}
	if _, err := os.Stat(filepath.Join(datasets, "demo-absent")); !os.IsNotExist(err) {
		t.Errorf("a refused demo name left a directory: %v", err)
	}
	rootless := newTestHandlerWithRepository(t, responseRecipeGenerator(t, &fakeGenerator{}))
	if status, body, _ := registerDemo(rootless, demoSets[0]); status != http.StatusNotImplemented {
		t.Errorf("a demo set on a server with no datasets directory: status=%d body=%s, want it to say it needs one", status, body)
	}
	// The shell that answers before any model is chosen registers a demo set the same way.
	idleStore, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer idleStore.Close()
	idle := &IdleShell{Repository: idleStore, Intake: LibraryIntake{DatasetRoot: t.TempDir()}}
	if status, body, receipt := registerDemo(idle, demoSets[0]); status != http.StatusOK || receipt.Name != demoSets[0] || receipt.Files == 0 {
		t.Errorf("a demo set while no model serves: status=%d body=%s, want it registered", status, body)
	}

	// The same route still imports a file by its path, and refuses a path that names none.
	source := filepath.Join(t.TempDir(), "chats.jsonl")
	if err := os.WriteFile(source, []byte(`{"instruction":"Add","input":"2 and 3","output":"5"}`+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	for name, scenario := range map[string]struct {
		path string
		want int
	}{
		"from-file": {source, http.StatusOK},
		"no-file":   {filepath.Join(filepath.Dir(source), "absent.jsonl"), http.StatusUnprocessableEntity},
	} {
		imported := serveTestRequest(handler, http.MethodPost, "/library/register", fmt.Sprintf(`{"kind":"dataset","name":%q,"path":%q}`, name, filepath.ToSlash(scenario.path)))
		if imported.Code != scenario.want {
			t.Errorf("importing %s by its path: status=%d body=%s, want %d", name, imported.Code, imported.Body.String(), scenario.want)
		}
	}
}

// byteVocabulary is a tokenizer.json vocabulary of one token for each byte a demo
// file may hold, a printable ASCII byte or a newline, spelled as the byte-level
// scheme spells it: a space and a newline as their own marker letters, every other
// byte as itself. Token 0 is left free for the token that ends a sequence.
func byteVocabulary() map[string]int {
	vocabulary := map[string]int{"Ġ": 1, "Ċ": 2}
	for letter := '!'; letter <= '~'; letter++ {
		vocabulary[string(letter)] = len(vocabulary) + 1
	}
	return vocabulary
}

// skipCounter is a reporter that counts the steps a run reports and keeps how
// many conversations the run said it skipped.
type skipCounter struct {
	stepCounter
	skipped *atomic.Int64
}

// Metric keeps the count of skipped conversations, the one metric a run reports
// only when it left some out.
func (counter skipCounter) Metric(metric operation.Metric) {
	if metric.Name == "skipped_conversations" {
		counter.skipped.Store(int64(metric.Value))
	}
}

// TestDemoDatasetsTrainOnTheTinyModel: each demo set, registered by its one
// request, fine-tunes a tiny served model from the Train tab's default
// capability. The model reads a token a byte, so it reads plain English; the run
// uses every conversation of the set, skipping none, and records the model and the
// set in and the trained model and its verdict out. What keeps that true of the
// files is held too: plain ASCII, no two records opening alike (a run holds
// conversations out by how they open), and each conversation within one window.
// Every reply of the identity set says the made-up name or where it was made.
func TestDemoDatasetsTrainOnTheTinyModel(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	root := t.TempDir()
	roots := dataroot.Roots{Models: filepath.Join(root, "models"), Datasets: filepath.Join(root, "datasets"), Checkpoints: filepath.Join(root, "checkpoints")}
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	vocabulary := byteVocabulary()
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: len(vocabulary) + 1, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 11,
	})
	served := filepath.Join(roots.Models, "served")
	model := writeWorkspaceModel(t, served, weights, shapes)
	// The helper's tokenizer knows four letters and its model declares no context or end
	// token; the demo sets need the byte vocabulary, a window that holds a conversation
	// and the token that separates the conversations packed into one.
	tokenizer, err := json.Marshal(map[string]any{"model": map[string]any{"type": "BPE", "vocab": vocabulary, "merges": []string{}}})
	if err != nil {
		t.Fatal(err)
	}
	config := fmt.Sprintf(`{"model_type":"llama","num_attention_heads":2,"head_dim":4,"max_position_embeddings":%d,"rope_theta":10000.0,"rms_norm_eps":1e-6,"tie_word_embeddings":true}`, demoContext)
	for name, document := range map[string]string{"tokenizer.json": string(tokenizer), "config.json": config, "generation_config.json": `{"eos_token_id":0}`} {
		if err := os.WriteFile(filepath.Join(served, name), []byte(document), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	handler := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{}), func(config *Config) {
		config.LibraryIntake.DatasetRoot = roots.Datasets
	})
	workspace, err := NewSFTWorkspace(store, roots, SFTTrainer{Model: model, File: served, Render: sftLetters})
	if err != nil {
		t.Fatal(err)
	}
	capabilities, err := workspace.WorkflowCapabilities(ctx, WorkflowTraining)
	if err != nil || len(capabilities) == 0 {
		t.Fatalf("training capabilities = %+v, %v, want fine-tuning the whole model first", capabilities, err)
	}
	for _, name := range demoSets {
		status, body, receipt := registerDemo(handler, name)
		if status != http.StatusOK {
			t.Fatalf("registering the demo set %s: status=%d body=%s", name, status, body)
		}
		records := demoRecords(t, name)
		opened := map[string]bool{}
		for index, record := range records {
			text := record.prompt() + record.Output
			if plain := strings.IndexFunc(text, func(letter rune) bool { return letter != '\n' && (letter < ' ' || letter > '~') }); plain >= 0 || record.Output == "" {
				t.Errorf("%s record %d holds a reply %q and, at %d, a byte the tiny model has no token for; a demo record is a reply in plain ASCII", name, index+1, record.Output, plain)
			}
			// A role's letter opens each of the two turns.
			if tokens := len(text) + len("bc"); tokens >= demoContext {
				t.Errorf("%s record %d is %d tokens, which a window of %d does not hold", name, index+1, tokens, demoContext)
			}
			if opened[record.prompt()] {
				t.Errorf("%s record %d opens as an earlier record does: %q", name, index+1, record.prompt())
			}
			opened[record.prompt()] = true
			if name == demoIdentity && !strings.Contains(record.Output, "Pebble") && !strings.Contains(record.Output, "overgo workbench") {
				t.Errorf("%s record %d replies %q, which says neither the made-up name Pebble nor that it was made in the overgo workbench", name, index+1, record.Output)
			}
		}
		steps, skipped := &atomic.Int64{}, &atomic.Int64{}
		input := json.RawMessage(fmt.Sprintf(`{"dataset":%q,"steps":%d,"output":%q}`, receipt.Dataset.String(), demoSteps, name+"-tuned"))
		completion, err := workspace.ExecuteWorkflow(ctx, WorkflowTraining, capabilities[0], input, skipCounter{stepCounter: stepCounter{steps: steps}, skipped: skipped})
		if err != nil || len(completion.Outputs) != 2 || steps.Load() != demoSteps || skipped.Load() != 0 {
			t.Fatalf("training on %s = (%+v, %v) after %d steps with %d conversations skipped, want the trained model and its verdict after %d steps on every conversation", name, completion, err, steps.Load(), skipped.Load(), demoSteps)
		}
		run, err := runrecord.RequireRun(ctx, store, completion.Run)
		if err != nil || run.Outcome != runrecord.OutcomeSucceeded || len(run.Inputs) != 2 || !slices.Contains(run.Inputs, model) || !slices.Contains(run.Inputs, receipt.Dataset) {
			t.Fatalf("the run recorded for %s = %+v (%v), want it succeeded with the model and the demo set in", name, run, err)
		}
		if _, err := os.Stat(filepath.Join(roots.Checkpoints, name+"-tuned")); err != nil {
			t.Fatalf("the model trained on %s was not written under the checkpoints root: %v", name, err)
		}
	}
}

// TestDemoDatasetsLeaveNoTemporaryFile: the importer reads a path, so a demo
// set's records pass through a file in the temporary directory. Whether the set
// registers, is asked for again, or is no demo set at all, the request
// leaves nothing in that directory; and where that directory is missing the
// request is refused with nothing written. The test names the temporary directory
// through the environment, so it does not run beside the others.
func TestDemoDatasetsLeaveNoTemporaryFile(t *testing.T) {
	temporary := t.TempDir()
	for _, variable := range []string{"TMPDIR", "TMP", "TEMP"} {
		t.Setenv(variable, temporary)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	datasets := t.TempDir()
	handler := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{}), func(config *Config) {
		config.LibraryIntake.DatasetRoot = datasets
	})
	for _, scenario := range []struct {
		name string
		want int
	}{
		{demoSets[0], http.StatusOK},
		{demoSets[0], http.StatusOK},
		{"demo-absent", http.StatusUnprocessableEntity},
	} {
		if status, body, _ := registerDemo(handler, scenario.name); status != scenario.want {
			t.Fatalf("registering %s: status=%d body=%s, want %d", scenario.name, status, body, scenario.want)
		}
		if left, err := os.ReadDir(temporary); err != nil || len(left) != 0 {
			t.Fatalf("registering %s, answered %d, left %v in the temporary directory (%v), want nothing", scenario.name, scenario.want, left, err)
		}
	}
	// Where the temporary directory is missing, the file cannot be made: the request is refused and no dataset is written.
	for _, variable := range []string{"TMPDIR", "TMP", "TEMP"} {
		t.Setenv(variable, filepath.Join(temporary, "absent"))
	}
	if status, body, _ := registerDemo(handler, demoSets[1]); status != http.StatusUnprocessableEntity {
		t.Errorf("registering %s with no temporary directory: status=%d body=%s, want it refused", demoSets[1], status, body)
	}
	if _, err := os.Stat(filepath.Join(datasets, demoSets[1])); !os.IsNotExist(err) {
		t.Errorf("a demo set refused for want of a temporary file left a directory: %v", err)
	}
}

// TestDemoDatasetsPageOffersEachSet holds the Datasets tab to the demo sets
// the server carries: the page names every set the server holds and no other,
// and registers one through the Library's route. A set added to the server
// with no button, or a button for a set that is not there, fails here. This
// reads the page's source; what a press does is held in the browser
// (TestWebUIBrowserDemoDatasets).
func TestDemoDatasetsPageOffersEachSet(t *testing.T) {
	t.Parallel()
	source, err := fs.ReadFile(webuiFS, "mod/datasets.js")
	if err != nil {
		t.Fatal(err)
	}
	carried, err := demoDatasets.ReadDir("demo")
	if err != nil {
		t.Fatal(err)
	}
	var held []string
	for _, entry := range carried {
		held = append(held, strings.TrimSuffix(entry.Name(), ".jsonl"))
	}
	var offered []string
	for _, named := range regexp.MustCompile(`\["(demo-[a-z-]+)", "Add the demo`).FindAllSubmatch(source, -1) {
		offered = append(offered, string(named[1]))
	}
	slices.Sort(held)
	slices.Sort(offered)
	if len(held) == 0 || !slices.Equal(held, offered) || !slices.Equal(held, slices.Sorted(slices.Values(demoSets))) {
		t.Errorf("the server carries the demo sets %v and the Datasets tab offers %v; want the page to offer each set the server carries and no other", held, offered)
	}
	if !strings.Contains(string(source), `overgo.api.post("/library/register", { kind: "dataset", demo: name }`) {
		t.Error("the Datasets tab does not register a demo set through the Library's route by the set's name")
	}
}
