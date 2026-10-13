package trainingworkflow

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
)

// pieces joins the text of the segments that are, or are not, learned.
func pieces(segments []conversationSegment, learned bool) string {
	var text strings.Builder
	for _, segment := range segments {
		if segment.Trained == learned {
			text.WriteString(segment.Text)
		}
	}
	return text.String()
}

// TestAssistantOnlyLoss runs one conversation dataset twice, learning the
// replies and then every token, and checks the replies-only run reads, trains
// and is judged on the replies alone: the records mark them, the held-out
// examples' targets are exactly their tokens, the two runs train to different
// models, and each publishes its own verdict, the held-out accuracy with the
// mask on and off.
func TestAssistantOnlyLoss(t *testing.T) {
	run := registerConversations(t, periodic(12))
	ctx := t.Context()
	var trained [2]string
	var verdicts [2]artifact.ID
	for index, prompts := range []bool{false, true} {
		output := filepath.Join(run.root, map[bool]string{false: "replies", true: "every-token"}[prompts])
		result, err := RunConversationSFT(ctx, run.store, ConversationSFT{
			Dataset: run.source, Render: letters, ModelDirectory: run.model, WorkDirectory: run.work,
			OutputDirectory: output, Steps: 6, Prompts: prompts,
		})
		if err != nil {
			t.Fatalf("prompts=%t: %v", prompts, err)
		}
		verdicts[index] = result.Verdict
		weights, err := filepath.Glob(filepath.Join(output, "*.safetensors"))
		if err != nil || len(weights) == 0 {
			t.Fatalf("prompts=%t: the run left no weights in %s (%v)", prompts, output, err)
		}
		data, err := os.ReadFile(weights[0])
		if err != nil {
			t.Fatal(err)
		}
		trained[index] = string(data)
		var verdict struct {
			BaselineValue  float64 `json:"baseline_value"`
			CandidateValue float64 `json:"candidate_value"`
		}
		content, found, err := artifact.ReadContent(ctx, run.store, result.Verdict)
		if err != nil || !found || json.Unmarshal(content.Data, &verdict) != nil {
			t.Fatalf("prompts=%t: verdict content = (found %t, %v)", prompts, found, err)
		}
		t.Logf("held-out accuracy with prompts=%t: base %.4f, trained %.4f", prompts, verdict.BaselineValue, verdict.CandidateValue)
	}
	if trained[0] == trained[1] {
		t.Fatal("learning the replies and learning every token trained to the same model: the mask changed nothing")
	}
	if verdicts[0] == verdicts[1] {
		t.Fatal("the two runs share one verdict")
	}

	// Each run's held-out examples are judged on the tokens it learns.
	tokens, err := loadTrainableTokenizerFor(run.model, true)
	if err != nil {
		t.Fatal(err)
	}
	for _, prompts := range []bool{false, true} {
		records, skipped, err := conversationRecords(ctx, run.store, run.source, letters, prompts, tokens)
		if err != nil || len(skipped) != 0 {
			t.Fatalf("records = %d, skipped %v, err %v: want every conversation usable", len(records), skipped, err)
		}
		registered, err := registerRecords(ctx, run.store, run.work, records)
		if err != nil {
			t.Fatal(err)
		}
		objective, err := LoadObjectiveAlias(ctx, run.store, conversationAliasPrefix+registered.Dataset.DigestHex()[:conversationNameDigits])
		if err != nil {
			t.Fatal(err)
		}
		data, held, err := textObjectiveData(ctx, run.store, objective, tokens, conversationWindow, true, 0)
		if err != nil {
			t.Fatal(err)
		}
		for id, example := range data.Examples {
			learned := len(example.Target)
			read := len(held[id].tokens) - 1
			if prompts && learned != read || !prompts && !(0 < learned && learned < read) {
				t.Errorf("prompts=%t: record %s is judged on %d of its %d predictable tokens", prompts, id, learned, read)
			}
			// The conversations speak a and b as the user and c and d as the assistant.
			if text := learnedText(t, held[id], tokens); !prompts && (text == "" || strings.Trim(text, "cd") != "") {
				t.Errorf("record %s learns %q, want the reply's letters alone", id, text)
			}
		}
	}
}

// learnedText spells the tokens a record learns.
func learnedText(t *testing.T, record textRecord, tokens trainableTokenizer) string {
	t.Helper()
	var text strings.Builder
	for index, id := range record.tokens {
		if record.trained[index] {
			text.WriteString(tokens.piece(id))
		}
	}
	return text.String()
}

// TestSegmentConversationCutsAtTheReply checks a conversation is cut where each
// assistant reply starts and ends: the generation prompt that introduces a
// reply, the user's turns and the template's markers are read, the reply's
// own text is learned, and learning the prompts makes the whole one segment.
func TestSegmentConversationCutsAtTheReply(t *testing.T) {
	messages := []dataset.ChatMessage{
		{Role: "system", Content: "ab"}, {Role: "user", Content: "ba"}, {Role: "assistant", Content: "cd"},
		{Role: "user", Content: "a"}, {Role: "assistant", Content: "dc"},
	}
	segments, err := segmentConversation(messages, letters, false)
	if err != nil {
		t.Fatal(err)
	}
	if learned := pieces(segments, true); learned != "cddc" {
		t.Errorf("learned %q, want the two replies, cd and dc", learned)
	}
	whole, _ := letters(messages, false)
	if text := pieces(segments, true) + pieces(segments, false); len(text) != len(whole) {
		t.Errorf("the segments hold %d characters, the conversation renders %d", len(text), len(whole))
	}
	var spelled strings.Builder
	for _, segment := range segments {
		spelled.WriteString(segment.Text)
	}
	if spelled.String() != whole {
		t.Errorf("segments spell %q, want the rendering %q", spelled.String(), whole)
	}
	// The assistant's own letter, the generation prompt serving supplies, is read, not learned.
	if segments[0].Trained || !strings.HasSuffix(segments[0].Text, assistantLetter) {
		t.Errorf("first segment %+v: want the prompt through the assistant's letter, read", segments[0])
	}
	every, err := segmentConversation(messages, letters, true)
	if err != nil || len(every) != 1 || !every[0].Trained || every[0].Text != whole {
		t.Errorf("learning the prompts gave %+v (%v), want the whole rendering as one learned segment", every, err)
	}
}

// TestSegmentConversationRefusesWhatItCannotCut checks a conversation with no
// reply, and a template whose renderings do not extend one another, are
// refused with the way out, rather than cut at the wrong place.
func TestSegmentConversationRefusesWhatItCannotCut(t *testing.T) {
	if _, err := segmentConversation([]dataset.ChatMessage{{Role: "user", Content: "ab"}}, letters, false); err == nil || !strings.Contains(err.Error(), "no assistant reply") {
		t.Errorf("a conversation with no reply: err = %v", err)
	}
	// A template that drops what an earlier assistant said, as one that hides its reasoning does.
	forgetful := func(messages []dataset.ChatMessage, generationPrompt bool) (string, error) {
		var kept []dataset.ChatMessage
		for index, message := range messages {
			if message.Role == "assistant" && index < len(messages)-1 {
				message.Content = ""
			}
			kept = append(kept, message)
		}
		return letters(kept, generationPrompt)
	}
	messages := []dataset.ChatMessage{
		{Role: "user", Content: "ab"}, {Role: "assistant", Content: "cd"}, {Role: "user", Content: "ba"}, {Role: "assistant", Content: "dc"},
	}
	if _, err := segmentConversation(messages, forgetful, false); err == nil || !strings.Contains(err.Error(), "turn by turn") {
		t.Errorf("a template that rewrites earlier turns: err = %v", err)
	}
	if segments, err := segmentConversation(messages, forgetful, true); err != nil || len(segments) != 1 {
		t.Errorf("learning every token needs no cut: %+v (%v)", segments, err)
	}
}

// bytesOf is a tokenizer's byte-per-token encoding of text.
func bytesOf(text string) []int {
	ids := make([]int, len(text))
	for index := range text {
		ids[index] = int(text[index])
	}
	return ids
}

// TestEncodeConversationMarksTheTokensOfLearnedSegments checks each token takes
// its segment's flag, the start token opens the conversation once, and a
// tokenizer that disagrees with itself about where a reply starts is refused.
func TestEncodeConversationMarksTheTokensOfLearnedSegments(t *testing.T) {
	const start = 99
	tokens := trainableTokenizer{
		encode: func(text string) ([]int, error) { return append([]int{start}, bytesOf(text)...), nil },
	}
	segments := []conversationSegment{{Text: "ab"}, {Text: "cd", Trained: true}, {Text: "e"}}
	ids, trained, err := encodeConversation(tokens, segments)
	if err != nil {
		t.Fatal(err)
	}
	if want := append([]int{start}, bytesOf("abcde")...); !slices.Equal(ids, want) {
		t.Errorf("tokens %v, want %v: the start token opens the conversation once", ids, want)
	}
	if want := []bool{false, false, false, true, true, false}; !slices.Equal(trained, want) {
		t.Errorf("trained %v, want %v", trained, want)
	}
	// A tokenizer that merges "bc" into one token, so the boundary after "ab" falls inside it.
	merging := trainableTokenizer{encode: func(text string) ([]int, error) {
		var ids []int
		for position := 0; position < len(text); position++ {
			if strings.HasPrefix(text[position:], "bc") {
				ids, position = append(ids, 200), position+1
				continue
			}
			ids = append(ids, int(text[position]))
		}
		return ids, nil
	}}
	if _, _, err := encodeConversation(merging, segments); err == nil || !strings.Contains(err.Error(), "train on the prompts too") {
		t.Errorf("a tokenizer that merges across a boundary: err = %v", err)
	}
}

// TestConversationBoundaryHoldsForAPrefixingTokenizer holds a segment boundary to what the
// tokenizer reads the whole text as: a vocabulary that adds a space at the start of the text
// it encodes, as a sentencepiece one does, and one that adds an end token, cut a conversation
// at the tokens its whole encoding has (a continuation encoder re-added the space at every
// boundary, so every conversation was refused); and a reply's learned tokens end at the
// template's end-of-turn marker, not at the newline the template writes after it.
func TestConversationBoundaryHoldsForAPrefixingTokenizer(t *testing.T) {
	const start, space, end = 99, 32, 98
	segments := []conversationSegment{{Text: "ab"}, {Text: "cd", Trained: true}, {Text: "e"}}
	prefixing := trainableTokenizer{encode: func(text string) ([]int, error) { return append([]int{start, space}, bytesOf(text)...), nil }}
	ids, trained, err := encodeConversation(prefixing, segments)
	if want := append([]int{start, space}, bytesOf("abcde")...); err != nil || !slices.Equal(ids, want) {
		t.Fatalf("a prefixing tokenizer: tokens %v (%v), want its whole encoding %v", ids, err, want)
	}
	if want := []bool{false, false, false, false, true, true, false}; !slices.Equal(trained, want) {
		t.Errorf("a prefixing tokenizer: trained %v, want %v: the prefix belongs to the first segment", trained, want)
	}
	closing := trainableTokenizer{end: end, ended: true, encode: func(text string) ([]int, error) { return append(bytesOf(text), end), nil }}
	ids, trained, err = encodeConversation(closing, segments)
	if want := append(bytesOf("abcde"), end); err != nil || !slices.Equal(ids, want) {
		t.Fatalf("a tokenizer that adds an end token: tokens %v (%v), want %v", ids, err, want)
	}
	if want := []bool{false, false, true, true, false, false}; !slices.Equal(trained, want) {
		t.Errorf("a tokenizer that adds an end token: trained %v, want %v: the end token follows the last segment", trained, want)
	}

	// A template that closes each turn with a marker and a newline.
	marked := func(messages []dataset.ChatMessage, generationPrompt bool) (string, error) {
		var text strings.Builder
		for _, message := range messages {
			text.WriteString("<" + message.Role + ">" + message.Content + "</>\n")
		}
		if generationPrompt {
			text.WriteString("<assistant>")
		}
		return text.String(), nil
	}
	messages := []dataset.ChatMessage{{Role: "user", Content: "ab"}, {Role: "assistant", Content: "cd"}}
	cut, err := segmentConversation(messages, marked, false)
	want := []conversationSegment{{Text: "<user>ab</>\n<assistant>"}, {Text: "cd</>", Trained: true}, {Text: "\n"}}
	if err != nil || !slices.Equal(cut, want) {
		t.Errorf("a template that follows its marker with a newline: segments %+v (%v), want %+v", cut, err, want)
	}
	// A template with no marker keeps whatever follows the reply as the reply's own.
	plain := func(messages []dataset.ChatMessage, generationPrompt bool) (string, error) {
		text, err := letters(messages, generationPrompt)
		if !generationPrompt {
			text += "\n"
		}
		return text, err
	}
	if cut, err = segmentConversation(messages, plain, false); err != nil || len(cut) == 0 || cut[len(cut)-1].Text != "cd\n" || !cut[len(cut)-1].Trained {
		t.Errorf("a template with no marker: segments %+v (%v), want the reply learned through its newline", cut, err)
	}
}

// TestPackWindowsCarriesTheMask checks a window's flags are its tokens' flags
// as the stream laid them, the end token between documents is learned only
// after a document learned throughout, and a window with nothing to learn is
// refused.
func TestPackWindowsCarriesTheMask(t *testing.T) {
	const end = 0
	documents := [][]int{{1, 2, 3}, {4, 5}}
	flags := [][]bool{{false, true, true}, {false, true}}
	windows, trained, _, err := packWindows(documents, flags, end, 4, 2)
	if err != nil {
		t.Fatal(err)
	}
	// The stream is 1 2 3 end 4 5 end, seven tokens, cycled.
	if want := [][]int{{1, 2, 3, end}, {4, 5, end, 1}}; !slices.EqualFunc(windows, want, slices.Equal) {
		t.Fatalf("windows %v, want %v", windows, want)
	}
	if want := [][]bool{{false, true, true, false}, {false, true, false, false}}; !slices.EqualFunc(trained, want, slices.Equal) {
		t.Fatalf("trained %v, want %v: the end token after a document with a masked token is read", trained, want)
	}
	_, whole, _, err := packWindows([][]int{{1, 2, 3}}, [][]bool{{true, true, true}}, end, 4, 1)
	if err != nil || !slices.Equal(whole[0], []bool{true, true, true, true}) {
		t.Fatalf("a document learned throughout: trained %v (%v), want its end token learned too", whole, err)
	}
	if _, _, _, err := packWindows([][]int{{1, 2, 3}}, [][]bool{{true, false, false}}, end, 2, 1); err == nil || !strings.Contains(err.Error(), "no token to learn") {
		t.Errorf("a window whose only learned token is its first: err = %v", err)
	}
	if _, none, _, err := packWindows(documents, nil, end, 4, 2); err != nil || none != nil {
		t.Errorf("no flags: trained %v (%v), want none", none, err)
	}
}

// TestHeldoutAnswersOnlyTheTokensItLearns checks a conversation record is
// answered at the positions whose next token it learns and nowhere else, as
// many answers as the targets the example is judged on.
func TestHeldoutAnswersOnlyTheTokensItLearns(t *testing.T) {
	run := registerConversations(t, periodic(2))
	model, err := loadTrainableModel(run.model)
	if err != nil {
		t.Fatal(err)
	}
	tokens := []int{1, 2, 3, 2, 1, 3}
	trained := []bool{false, false, false, true, true, false}
	example := HeldoutExample{Input: []float64{1, 2, 3, 2, 1, 3}}
	prediction, err := nextTokens(model, example, trained)
	if err != nil {
		t.Fatal(err)
	}
	if want := learnedTargets(tokens, trained); len(prediction.Values) != len(want) || !slices.Equal(want, []int{2, 1}) {
		t.Fatalf("%d answers for targets %v, want one answer for each of the two learned tokens", len(prediction.Values), want)
	}
	everything, err := nextTokens(model, example, nil)
	if err != nil || len(everything.Values) != len(tokens)-1 {
		t.Fatalf("every token: %d answers (%v), want %d", len(everything.Values), err, len(tokens)-1)
	}
}

// TestIdenticalAnswersAreRefusedPlainly runs an adapter for four steps over
// conversations whose replies it barely moves: the adapted model then answers
// every held-out token as the base does, the two reports are one document, and
// the run says so rather than failing as an invalid verdict.
func TestIdenticalAnswersAreRefusedPlainly(t *testing.T) {
	run := registerConversations(t, periodic(12))
	_, err := RunConversationSFT(t.Context(), run.store, ConversationSFT{
		Dataset: run.source, Render: letters, ModelDirectory: run.model, WorkDirectory: run.work,
		OutputDirectory: filepath.Join(run.root, "trained"), Steps: 4, LoRA: &LoRA{Rank: 2, Alpha: 4},
	})
	if err == nil || !strings.Contains(err.Error(), "exactly as the base model does") {
		t.Fatalf("an adapter that answers as the base does: err = %v, want the plain refusal", err)
	}
	if strings.Contains(err.Error(), "invalid held-out verdict") {
		t.Fatalf("err = %v: the refusal reached the verdict's own validation", err)
	}
}
