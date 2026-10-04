package trainingworkflow

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

// conversationRun is a tiny dense model, a store holding a registered
// conversation dataset, and the work and output directories a run needs.
type conversationRun struct {
	store  *overgodb.Store
	model  string
	source artifact.ID
	work   string
	root   string
}

// letters renders a conversation within the tiny model's four-letter
// vocabulary: each message is its role's letter followed by its content.
func letters(messages []dataset.ChatMessage) (string, error) {
	var text strings.Builder
	for _, message := range messages {
		text.WriteString(map[string]string{"system": "a", "user": "b", "assistant": "c"}[message.Role] + message.Content)
	}
	return text.String(), nil
}

func registerConversations(t *testing.T, conversations []dataset.Conversation) conversationRun {
	t.Helper()
	ctx := t.Context()
	root := t.TempDir()
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 5,
	})
	model := filepath.Join(root, "model")
	writeModel(t, model, weights, shapes)
	if err := os.WriteFile(filepath.Join(model, "generation_config.json"), []byte(`{"eos_token_id":0}`), 0o600); err != nil {
		t.Fatal(err)
	}
	directory := filepath.Join(root, "conversations")
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	for index, conversation := range conversations {
		data, err := json.Marshal(conversation)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(directory, fmt.Sprintf("%02d.json", index+1)), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Error(err)
		}
	})
	registered, err := dataset.RegisterDirectoryDatasetAs(ctx, store, "chats", directory, dataset.ModalityStructured)
	if err != nil {
		t.Fatal(err)
	}
	return conversationRun{store: store, model: model, source: registered.Dataset, work: filepath.Join(root, "work"), root: root}
}

// periodic returns count two-message conversations, each its own: the first
// six differ in length, and the next six repeat them from the other letter.
func periodic(count int) []dataset.Conversation {
	var conversations []dataset.Conversation
	for index := range count {
		user, assistant := strings.Repeat("ab", 4), strings.Repeat("cd", 4)
		if index >= 6 {
			user, assistant = strings.Repeat("ba", 4), strings.Repeat("dc", 4)
		}
		conversations = append(conversations, dataset.Conversation{Messages: []dataset.ChatMessage{
			{Role: "user", Content: user[:1+index%6]},
			{Role: "assistant", Content: assistant[:1+index%6]},
		}})
	}
	return conversations
}

// TestConversationSFTTrainsOnRenderedConversations fine-tunes a tiny dense
// model on twelve conversations: each renders through the supplied template
// into one text record of a registered dataset, one record is held out
// before the rest train, the objective publishes under an alias of its own
// beside the shared one, and the run publishes the trained model and a
// held-out verdict. Running the same conversations again registers the same
// dataset and objective rather than another.
func TestConversationSFTTrainsOnRenderedConversations(t *testing.T) {
	run := registerConversations(t, periodic(12))
	ctx := t.Context()
	var steps []int
	request := ConversationSFT{
		Dataset: run.source, Render: letters, ModelDirectory: run.model, WorkDirectory: run.work,
		OutputDirectory: filepath.Join(run.root, "trained-1"), Steps: 6,
		Step: func(step, total int, _ float64) { steps = append(steps, step*100+total) },
	}
	result, err := RunConversationSFT(ctx, run.store, request)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Trained.Valid() || !result.Verdict.Valid() {
		t.Fatalf("a conversation run published trained %v and verdict %v", result.Trained, result.Verdict)
	}
	if len(steps) != 6 || steps[0] != 106 || steps[5] != 606 {
		t.Fatalf("the run reported steps %v, want each of six", steps)
	}
	files, err := filepath.Glob(filepath.Join(run.work, "sft-*", "*.txt"))
	if err != nil || len(files) != 12 {
		t.Fatalf("rendered %d records (%v), want twelve", len(files), err)
	}
	first, err := os.ReadFile(files[0])
	want, _ := letters(periodic(12)[0].Messages)
	if err != nil || string(first) != want {
		t.Fatalf("first record = %q (%v), want the first conversation rendered as %q", first, err, want)
	}
	rendered, found, err := run.store.ResolveAlias(ctx, "dataset.registered."+filepath.Base(filepath.Dir(files[0])))
	if err != nil || !found {
		t.Fatalf("the rendered conversations registered no dataset (%t, %v)", found, err)
	}
	alias := conversationAliasPrefix + rendered.DigestHex()[:conversationNameDigits]
	objective, err := LoadObjectiveAlias(ctx, run.store, alias)
	if err != nil || objective.Kind != trainingprogram.ObjectiveTokenPrediction || objective.Metric != trainingprogram.MetricTokenAccuracy {
		t.Fatalf("objective = %+v (%v), want token prediction judged by token accuracy", objective, err)
	}
	training, found, err := dataset.LoadMembership(ctx, run.store, objective.Split)
	if err != nil || !found || len(training.Records) != 11 {
		t.Fatalf("training membership holds %d records (found %t, %v), want eleven with one held out", len(training.Records), found, err)
	}
	if _, found, err := run.store.ResolveAlias(ctx, "objective.registered.text-text"); err != nil || found {
		t.Fatalf("the shared text objective alias bound (%t, %v) by a conversation run", found, err)
	}
	request.OutputDirectory = filepath.Join(run.root, "trained-2")
	if _, err := RunConversationSFT(ctx, run.store, request); err != nil {
		t.Fatalf("the same conversations a second time: %v", err)
	}
	if again, err := LoadObjectiveAlias(ctx, run.store, alias); err != nil || again.ID != objective.ID {
		t.Fatalf("a second run left objective %v (%v), want the one it published first, %v", again.ID, err, objective.ID)
	}
}

// TestConversationSFTDropsRepeatedConversations: conversations that render
// the same register one record, so a dataset of six conversations, three of
// them one repeated, trains on four records with one held out.
func TestConversationSFTDropsRepeatedConversations(t *testing.T) {
	conversations := periodic(4)
	run := registerConversations(t, append(conversations, conversations[0], conversations[0]))
	result, err := RunConversationSFT(t.Context(), run.store, ConversationSFT{
		Dataset: run.source, Render: letters, ModelDirectory: run.model, WorkDirectory: run.work,
		OutputDirectory: filepath.Join(run.root, "trained"), Steps: 2,
	})
	if err != nil || !result.Verdict.Valid() {
		t.Fatalf("six conversations of four distinct ran as (%v, %v)", result.Verdict, err)
	}
	if files, err := filepath.Glob(filepath.Join(run.work, "sft-*", "*.txt")); err != nil || len(files) != 4 {
		t.Fatalf("rendered %d records (%v), want the four distinct ones", len(files), err)
	}
}

// TestConversationSFTRefusesWhatItCannotJudge: a dataset of one conversation
// has none to hold out, a renderer that fails or renders nothing names the
// conversation, and a run without a renderer does not start.
func TestConversationSFTRefusesWhatItCannotJudge(t *testing.T) {
	for _, scenario := range []struct {
		name          string
		conversations int
		render        ConversationRenderer
		want          string
	}{
		{"one conversation", 1, letters, "at least 2"},
		{"one distinct conversation", 1, letters, "1 distinct"},
		{"renderer fails", 4, func([]dataset.ChatMessage) (string, error) { return "", errors.New("no template") }, "no template"},
		{"renders nothing", 4, func([]dataset.ChatMessage) (string, error) { return "", nil }, "renders as no text"},
		{"no renderer", 4, nil, "needs a store"},
	} {
		run := registerConversations(t, periodic(scenario.conversations))
		_, err := RunConversationSFT(t.Context(), run.store, ConversationSFT{
			Dataset: run.source, Render: scenario.render, ModelDirectory: run.model, WorkDirectory: run.work,
			OutputDirectory: filepath.Join(run.root, "trained"), Steps: 2,
		})
		if err == nil || !strings.Contains(err.Error(), scenario.want) {
			t.Errorf("%s: err = %v, want it to say %q", scenario.name, err, scenario.want)
		}
	}
}

// TestPreviewConversationsShowsWhatARunTrains pages a conversation dataset as
// a run reads it: the count is the dataset's files, the page holds only the
// conversations asked for, each is the rendered text the tokenizer splits, and
// every token is marked trained because a run has no loss mask.
func TestPreviewConversationsShowsWhatARunTrains(t *testing.T) {
	run := registerConversations(t, periodic(12))
	preview, err := PreviewConversations(t.Context(), run.store, run.source, letters, run.model, 1, 2)
	if err != nil || preview.Records != 12 || preview.Position != 1 || len(preview.Page) != 2 || preview.Page[0].ID != "02.json" || preview.Page[1].ID != "03.json" {
		t.Fatalf("preview = %+v (%v), want twelve records paged from the second, two of them", preview, err)
	}
	for index, record := range preview.Page {
		want, _ := letters(periodic(12)[index+1].Messages)
		var text strings.Builder
		for _, token := range record.Sequences[0].Tokens {
			text.WriteString(token.Piece)
			if !token.Trained {
				t.Errorf("%s: token %q is masked, want every token trained", record.ID, token.Piece)
			}
		}
		if len(record.Sequences) != 1 || text.String() != want {
			t.Errorf("%s: tokens spell %q over %d sequences, want the rendered conversation %q", record.ID, text.String(), len(record.Sequences), want)
		}
	}
	past, err := PreviewConversations(t.Context(), run.store, run.source, letters, run.model, 20, 2)
	if err != nil || past.Records != 12 || len(past.Page) != 0 {
		t.Fatalf("a page past the end = %+v (%v), want the count and no records", past, err)
	}
	if _, err := PreviewConversations(t.Context(), run.store, run.source, letters, run.model, 0, 0); err == nil {
		t.Fatal("an empty page was accepted")
	}
}
