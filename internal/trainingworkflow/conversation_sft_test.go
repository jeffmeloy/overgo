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

// assistantLetter is the assistant's role letter in the test template, which a
// generation prompt is.
const assistantLetter = "c"

// letters renders a conversation within the tiny model's four-letter
// vocabulary: each message is its role's letter followed by its content, and
// a generation prompt is the assistant's letter.
func letters(messages []dataset.ChatMessage, generationPrompt bool) (string, error) {
	var text strings.Builder
	for _, message := range messages {
		text.WriteString(map[string]string{"system": "a", "user": "b", "assistant": assistantLetter}[message.Role] + message.Content)
	}
	if generationPrompt {
		text.WriteString(assistantLetter)
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
	if err != nil {
		t.Fatal(err)
	}
	var record conversationRecord
	if err := json.Unmarshal(first, &record); err != nil {
		t.Fatalf("first record %q is not a conversation record: %v", first, err)
	}
	want, _ := letters(periodic(12)[0].Messages, false)
	var text, learned strings.Builder
	for _, segment := range record.Segments {
		text.WriteString(segment.Text)
		if segment.Trained {
			learned.WriteString(segment.Text)
		}
	}
	if text.String() != want || learned.String() != periodic(12)[0].Messages[1].Content {
		t.Fatalf("first record spells %q and learns %q, want the first conversation rendered as %q and its reply learned", text.String(), learned.String(), want)
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

// TestConversationRunSkipsAnUnusableConversationAndSaysSo holds a run to the records it
// can learn from: a conversation with no assistant reply and one whose reply lies
// beyond the window are each left out, named with the reason in the result and on the
// run's progress, the run trains and is judged on the six that remain; a dataset none
// of whose conversations can be learned from is refused, with the first reason.
func TestConversationRunSkipsAnUnusableConversationAndSaysSo(t *testing.T) {
	userOnly := dataset.Conversation{Messages: []dataset.ChatMessage{{Role: "user", Content: "ab"}}}
	overlong := dataset.Conversation{Messages: []dataset.ChatMessage{
		{Role: "user", Content: strings.Repeat("ab", conversationWindow)},
		{Role: "assistant", Content: "cd"},
	}}
	run := registerConversations(t, append(periodic(6), userOnly, overlong))
	var progress strings.Builder
	result, err := RunConversationSFT(t.Context(), run.store, ConversationSFT{
		Dataset: run.source, Render: letters, ModelDirectory: run.model, WorkDirectory: run.work,
		OutputDirectory: filepath.Join(run.root, "trained"), Steps: 2, Progress: &progress,
	})
	if err != nil || !result.Verdict.Valid() {
		t.Fatalf("a dataset of six usable conversations and two unusable ran as (%v, %v)", result.Verdict, err)
	}
	if len(result.Skipped) != 2 || result.Skipped[0].File != "07.json" || !strings.Contains(result.Skipped[0].Reason, "no assistant reply") ||
		result.Skipped[1].File != "08.json" || !strings.Contains(result.Skipped[1].Reason, "learns no token") {
		t.Fatalf("skipped = %+v, want the user-only conversation and the overlong one, each with its reason", result.Skipped)
	}
	if !strings.Contains(progress.String(), "skipped 2 of 8 conversations") {
		t.Errorf("progress %q does not say two of eight were skipped", progress.String())
	}
	if files, err := filepath.Glob(filepath.Join(run.work, "sft-*", "*.txt")); err != nil || len(files) != 6 {
		t.Errorf("rendered %d records (%v), want the six usable ones", len(files), err)
	}

	none := registerConversations(t, []dataset.Conversation{userOnly, {Messages: []dataset.ChatMessage{{Role: "user", Content: "ba"}}}})
	_, err = RunConversationSFT(t.Context(), none.store, ConversationSFT{
		Dataset: none.source, Render: letters, ModelDirectory: none.model, WorkDirectory: none.work,
		OutputDirectory: filepath.Join(none.root, "trained"), Steps: 2,
	})
	if err == nil || !strings.Contains(err.Error(), "the 2 skipped as unusable, the first 01.json") || !strings.Contains(err.Error(), "no assistant reply") {
		t.Fatalf("a dataset of unusable conversations: %v, want it refused naming the skipped and the first reason", err)
	}
}

// TestConversationsSharingAnOpeningNeverSplitAcrossTheVerdict holds the held-out side of a
// conversation run to whole groups: conversations that open alike (the same system prompt and
// first turns, whatever they go on to say) are one group, and held-out groups are taken whole, so
// no variant of a conversation the model trains on is one it is judged on, which the verdict would
// read as memorization; a dataset that opens one way has no held-out side to give and is refused.
func TestConversationsSharingAnOpeningNeverSplitAcrossTheVerdict(t *testing.T) {
	source, err := artifact.IdentifyBytes(artifact.KindDataset, []byte("chats"))
	if err != nil {
		t.Fatal(err)
	}
	record := func(opening, reply string) []byte {
		data, err := json.Marshal(conversationRecord{Segments: []conversationSegment{
			{Text: opening}, {Text: reply, Trained: true},
		}})
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	// Twenty conversations in ten openings, two replies to each: a tenth held out is two records.
	var records [][]byte
	for index := range 10 {
		opening := strings.Repeat("a", 1+index)
		records = append(records, record(opening, "cd"), record(opening, "dc"))
	}
	groups, err := conversationGroups(records)
	if err != nil {
		t.Fatal(err)
	}
	for index := 0; index < len(records); index += 2 {
		if groups[index] != groups[index+1] {
			t.Fatalf("records %d and %d open alike and are in groups %.12s and %.12s", index, index+1, groups[index], groups[index+1])
		}
	}
	_, training, err := conversationSplit(source, groups)
	if err != nil {
		t.Fatal(err)
	}
	trained := map[string]bool{}
	for _, member := range training.Records {
		trained[member.ID] = true
	}
	if len(trained) == 0 || len(trained) >= len(records) {
		t.Fatalf("training holds %d of %d records, want some held out and some kept", len(trained), len(records))
	}
	for index := 0; index < len(records); index += 2 {
		first := trained[dataset.RecordID(source, dataset.InventoryAssetName, uint64(index))]
		second := trained[dataset.RecordID(source, dataset.InventoryAssetName, uint64(index+1))]
		if first != second {
			t.Errorf("records %d and %d open alike but only one is trained on (%t, %t): a variant of what the model trained on is judged", index, index+1, first, second)
		}
	}

	// One opening for every conversation leaves nothing to hold out apart from the rest.
	alike, err := conversationGroups([][]byte{record("ab", "cd"), record("ab", "dc")})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := conversationSplit(source, alike); err == nil || !strings.Contains(err.Error(), "distinct way") {
		t.Errorf("conversations that all open one way: err = %v, want them refused", err)
	}
}

// TestObjectiveSFTAliasNeedsItsConversations holds a conversation objective to the run
// that reads its segments: trained as raw text, as the command line trains an objective
// by its alias, it would learn every token of the records' JSON and publish a verdict
// that looks like a masked run's, so the text run refuses it where a conversation run
// of the same objective, which published it, goes on.
func TestObjectiveSFTAliasNeedsItsConversations(t *testing.T) {
	run := registerConversations(t, periodic(4))
	ctx := t.Context()
	if _, err := RunConversationSFT(ctx, run.store, ConversationSFT{
		Dataset: run.source, Render: letters, ModelDirectory: run.model, WorkDirectory: run.work,
		OutputDirectory: filepath.Join(run.root, "masked"), Steps: 2,
	}); err != nil {
		t.Fatal(err)
	}
	files, err := filepath.Glob(filepath.Join(run.work, "sft-*", "*.txt"))
	if err != nil || len(files) == 0 {
		t.Fatalf("the conversation run rendered %d records (%v)", len(files), err)
	}
	rendered, found, err := run.store.ResolveAlias(ctx, "dataset.registered."+filepath.Base(filepath.Dir(files[0])))
	if err != nil || !found {
		t.Fatalf("the rendered conversations registered no dataset (%t, %v)", found, err)
	}
	alias := conversationAliasPrefix + rendered.DigestHex()[:conversationNameDigits]
	_, err = RunTextHeldout(ctx, run.store, TextHeldoutRequest{
		Alias: alias, ModelDirectory: run.model, OutputDirectory: filepath.Join(run.root, "raw"), Steps: 2, Host: true,
	})
	if err == nil || !strings.Contains(err.Error(), "conversation objective") {
		t.Fatalf("an objective of conversations trained as raw text: %v, want it refused as a conversation objective", err)
	}
	if _, statErr := os.Stat(filepath.Join(run.root, "raw")); statErr == nil {
		t.Error("the refused run published a model")
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
		{"renderer fails", 4, func([]dataset.ChatMessage, bool) (string, error) { return "", errors.New("no template") }, "no template"},
		{"renders nothing", 4, func([]dataset.ChatMessage, bool) (string, error) { return "", nil }, "renders as no text"},
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
// a token is marked trained when the run learns it: the reply's alone, or
// every token when the run trains on prompts.
func TestPreviewConversationsShowsWhatARunTrains(t *testing.T) {
	run := registerConversations(t, periodic(12))
	preview, err := PreviewConversations(t.Context(), run.store, run.source, letters, run.model, false, 1, 2)
	if err != nil || preview.Records != 12 || preview.Position != 1 || len(preview.Page) != 2 || preview.Page[0].ID != "02.json" || preview.Page[1].ID != "03.json" {
		t.Fatalf("preview = %+v (%v), want twelve records paged from the second, two of them", preview, err)
	}
	for index, record := range preview.Page {
		messages := periodic(12)[index+1].Messages
		want, _ := letters(messages, false)
		var text, learned strings.Builder
		for _, token := range record.Sequences[0].Tokens {
			text.WriteString(token.Piece)
			if token.Trained {
				learned.WriteString(token.Piece)
			}
		}
		if len(record.Sequences) != 1 || text.String() != want {
			t.Errorf("%s: tokens spell %q over %d sequences, want the rendered conversation %q", record.ID, text.String(), len(record.Sequences), want)
		}
		if learned.String() != messages[1].Content {
			t.Errorf("%s: trained tokens spell %q, want the reply %q alone", record.ID, learned.String(), messages[1].Content)
		}
	}
	every, err := PreviewConversations(t.Context(), run.store, run.source, letters, run.model, true, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	for _, token := range every.Page[0].Sequences[0].Tokens {
		if !token.Trained {
			t.Errorf("token %q is masked though the run trains on prompts", token.Piece)
		}
	}
	past, err := PreviewConversations(t.Context(), run.store, run.source, letters, run.model, false, 20, 2)
	if err != nil || past.Records != 12 || len(past.Page) != 0 {
		t.Fatalf("a page past the end = %+v (%v), want the count and no records", past, err)
	}
	if _, err := PreviewConversations(t.Context(), run.store, run.source, letters, run.model, false, 0, 0); err == nil {
		t.Fatal("an empty page was accepted")
	}
}
