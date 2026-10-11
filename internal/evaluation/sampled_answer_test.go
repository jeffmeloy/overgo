package evaluation

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/inference"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// sampledVocab is a small vocabulary that merges: "A" and "n" are pieces and
// so is "An", so the text of the two sampled apart tokenizes to the one.
func sampledVocab(t *testing.T) *tokenizer.Vocab {
	t.Helper()
	tokens := []string{"<unk>", "<s>", "</s>", "▁", "A", "n", "s", "w", "e", "r", ":", "An", "Ans", "Answ", "Answe", "Answer", "Answer:", "▁A"}
	scores, types := make([]float32, len(tokens)), make([]int32, len(tokens))
	for index := range tokens {
		types[index] = int32(tokenizer.TokenNormal)
		if index >= 11 {
			scores[index] = float32(index)
		}
	}
	types[0], types[1], types[2] = int32(tokenizer.TokenUnknown), int32(tokenizer.TokenControl), int32(tokenizer.TokenControl)
	vocab, err := tokenizer.Load(&gguf.File{Metadata: []gguf.Metadata{
		testutil.GGUFScalar("tokenizer.ggml.model", gguf.ValueTypeString, "llama"),
		testutil.GGUFScalar("tokenizer.ggml.bos_token_id", gguf.ValueTypeUint32, uint32(1)),
		testutil.GGUFScalar("tokenizer.ggml.eos_token_id", gguf.ValueTypeUint32, uint32(2)),
		testutil.GGUFScalar("tokenizer.ggml.unknown_token_id", gguf.ValueTypeUint32, uint32(0)),
		testutil.GGUFScalar("tokenizer.ggml.add_bos_token", gguf.ValueTypeBool, true),
		testutil.GGUFScalar("tokenizer.ggml.add_space_prefix", gguf.ValueTypeBool, false),
		testutil.GGUFArray("tokenizer.ggml.tokens", gguf.ValueTypeString, tokens),
		testutil.GGUFArray("tokenizer.ggml.scores", gguf.ValueTypeFloat32, scores),
		testutil.GGUFArray("tokenizer.ggml.token_type", gguf.ValueTypeInt32, types),
	}})
	if err != nil {
		t.Fatal(err)
	}
	return vocab
}

// scriptedSampler is a generator that samples what it is told to: for each
// call the ids and the probability it reports for each, as far as the output
// limit, an ending id or a stop text. It returns the prompt's ids and those,
// less dropped of them, and counts its calls.
type scriptedSampler struct {
	vocab   *tokenizer.Vocab
	ids     []tokenizer.TokenID
	chances []float64
	dropped int
	calls   int
	asked   inference.GenerateOptions
}

func (s *scriptedSampler) Generate(_ context.Context, _ string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	s.calls, s.asked = s.calls+1, options
	ids := slices.Clone(options.PromptTokenIDs)
	var text strings.Builder
	for index, id := range s.ids {
		if index == options.MaxNewTokens {
			break
		}
		piece, err := s.vocab.DecodePiece(id, false)
		if err != nil {
			return nil, "", err
		}
		if err := options.OnToken(inference.TokenEvent{ID: id, Piece: piece, Index: index, SelectedProbability: s.chances[index]}); err != nil {
			return nil, "", err
		}
		ids = append(ids, id)
		text.WriteString(piece)
		if s.vocab.IsEOG(id) || slices.ContainsFunc(options.StopSequences, func(stop string) bool { return strings.HasSuffix(text.String(), stop) }) {
			break
		}
	}
	return ids[:len(ids)-s.dropped], text.String(), nil
}

// TestSampledAnswerKeepsItsTokensAndProbabilities holds a sampled answer to
// what was sampled, and its record to its draw.
//
// The ids kept are the ids sampled, with the logarithm of the probability
// each was drawn with beside it, one an id; the prompt's ids are the group's
// own; and what training reads of a draw is those ids with the completion
// marked, never the text tokenized again: here the text of two pieces sampled
// apart tokenizes to one merged piece, other ids than were sampled. A draw
// that reached its limit, one that ended at an ending id and one a stop text
// ended each say so, and one that ended for no reason named, one whose ids are
// not the prompt's and those sampled, and one sampled with no probability
// reported are refused and not recorded.
//
// A draw is recorded once: asked again, by this handle or one opened later,
// it is returned as recorded and nothing is sampled; a second writer of the
// same draw with the same content changes nothing, and one with other content
// is refused. Its seed comes of its group and its index, the same in any
// store. A group under other weights, another update of them, another
// template, sampler setting, code or prompt names other draws; before the
// first update the weights are the checkpoint's own.
//
// What the ids and probabilities cost is measured: bytes a sampled token.
func TestSampledAnswerKeepsItsTokensAndProbabilities(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	vocab := sampledVocab(t)
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	checkpoint := testutil.ArtifactID(t, artifact.KindCheckpoint, "sampled fixture checkpoint")
	training := testutil.ArtifactID(t, artifact.KindProfile, "sampled fixture training")
	group := sampledGroup{
		Group: "prompt 12 at update 0", Code: "inference surface 1", Weights: checkpoint, Tokenizer: testutil.ArtifactID(t, artifact.KindTokenizer, "sampled fixture tokenizer"),
		Template: "chat-template", Prompt: []tokenizer.TokenID{1, 16}, Ends: []tokenizer.TokenID{2}, Temperature: 0.7, MaxTokens: 4, Seed: 3,
	}
	held, err := loadSampled(ctx, store, group)
	if err != nil {
		t.Fatal(err)
	}
	apart := []tokenizer.TokenID{4, 5, 2}
	sampler := &scriptedSampler{vocab: vocab, ids: apart, chances: []float64{0.5, 0.25, 1}}
	first, err := group.sample(ctx, sampler, held, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(first.Completion, apart) || !slices.Equal(first.LogProbabilities, []float64{math.Log(0.5), math.Log(0.25), 0}) || first.End != sampledEndOfSequence || first.Text != "An" || first.Draw != 0 {
		t.Fatalf("the first draw is recorded as %+v, want the ids %v with their log probabilities, ended at an ending id", first, apart)
	}
	if asked := sampler.asked; !slices.Equal(asked.PromptTokenIDs, group.Prompt) || asked.Sampler == nil || asked.MaxNewTokens != group.MaxTokens || asked.PostSamplingProbabilities == 0 {
		t.Fatalf("the generator was asked %+v, want the group's prompt ids, a sampler, its limit and each id's probability", asked)
	}
	again, err := vocab.Encode(first.Text, tokenizer.EncodeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if ids, completion := group.tokens(first); slices.Equal(again, apart[:2]) || !slices.Equal(ids, []tokenizer.TokenID{1, 16, 4, 5, 2}) || !slices.Equal(completion, []bool{false, false, true, true, true}) {
		t.Fatalf("the text %q tokenizes to %v and the draw is read as %v with completion %v; want the text's ids other than the %v sampled, and the draw's the sampled ones after the prompt's", first.Text, again, ids, completion, apart[:2])
	}

	// The limit, and a stop text, which is another group's setting.
	sampler.ids, sampler.chances = []tokenizer.TokenID{4, 5, 6, 7, 8}, []float64{0.5, 0.5, 0.5, 0.5, 0.5}
	if limited, err := group.sample(ctx, sampler, held, 1); err != nil || limited.End != sampledOutputLimit || len(limited.Completion) != group.MaxTokens || limited.Seed == first.Seed {
		t.Fatalf("a draw of more ids than the limit = %+v (%v), want %d ids ended at the limit under a seed of its own", limited, err, group.MaxTokens)
	}
	stopping := group
	stopping.Stops = []string{"An"}
	stopped, err := loadSampled(ctx, store, stopping)
	if err != nil || stopped.observed == held.observed || len(stopped.values) != 0 {
		t.Fatalf("a group with a stop text holds %d draws under %s (%v), want none under another identity than %s", len(stopped.values), stopped.observed, err, held.observed)
	}
	if ended, err := stopping.sample(ctx, sampler, stopped, 0); err != nil || ended.End != sampledStopSequence || len(ended.Completion) != 2 || ended.Seed == first.Seed {
		t.Fatalf("a draw a stop text ended = %+v (%v), want two ids ended at the stop text, under another seed than the first group's", ended, err)
	}

	// Draws that are refused leave nothing: the next draw is still the third.
	for name, broken := range map[string]scriptedSampler{
		"that ends for no reason named":        {ids: []tokenizer.TokenID{4, 5}, chances: []float64{0.5, 0.5}},
		"whose ids are not those sampled":      {ids: apart, chances: []float64{0.5, 0.5, 1}, dropped: 1},
		"sampled with no probability reported": {ids: apart, chances: []float64{0.5, 0, 1}},
	} {
		broken.vocab = vocab
		if _, err := group.sample(ctx, &broken, held, 2); err == nil || len(held.values) != 2 {
			t.Fatalf("a draw %s = %v with %d draws held, want it refused and two held", name, err, len(held.values))
		}
	}

	// Recorded once: the same handle, a later one, and a store that never held it.
	if err := held.write(ctx); err != nil {
		t.Fatal(err)
	}
	calls := sampler.calls
	later, err := loadSampled(ctx, store, group)
	if err != nil || len(later.values) != 2 {
		t.Fatalf("a handle opened later holds %d draws (%v), want the two recorded", len(later.values), err)
	}
	for _, handle := range []*answers[sampledAnswer]{held, later} {
		if replayed, err := group.sample(ctx, sampler, handle, 0); err != nil || !reflect.DeepEqual(replayed, first) || sampler.calls != calls {
			t.Fatalf("the first draw asked again = %+v (%v) after %d more calls, want the record %+v and nothing sampled", replayed, err, sampler.calls-calls, first)
		}
	}
	elsewhere, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = elsewhere.Close() })
	fresh, err := loadSampled(ctx, elsewhere, group)
	if err != nil {
		t.Fatal(err)
	}
	sampler.ids, sampler.chances = apart, []float64{0.5, 0.25, 1}
	if same, err := group.sample(ctx, sampler, fresh, 0); err != nil || !reflect.DeepEqual(same, first) {
		t.Fatalf("the same draw in another store = %+v (%v), want %+v: its seed is its group's and its index's", same, err, first)
	}

	// Two writers of one draw: the same content changes nothing, other content is refused and the first stands.
	racing := group
	racing.Group = "prompt 13 at update 0"
	writers := make([]*answers[sampledAnswer], 3)
	for index := range writers {
		if writers[index], err = loadSampled(ctx, store, racing); err != nil {
			t.Fatal(err)
		}
	}
	for index, script := range [][]tokenizer.TokenID{apart, apart, {5, 4, 2}} {
		sampler.ids = script
		if _, err := racing.sample(ctx, sampler, writers[index], 0); err != nil {
			t.Fatal(err)
		}
		if err := writers[index].write(ctx); (err != nil) != (index == 2) || index == 2 && !errors.Is(err, overgodb.ErrBatchKeyConflict) {
			t.Fatalf("writer %d of the same draw = %v, want the first two written and the third, of other ids, refused", index, err)
		}
	}
	if settled, err := loadSampled(ctx, store, racing); err != nil || len(settled.values) != 1 || !slices.Equal(settled.values[0].Completion, apart) {
		t.Fatalf("the draw two writers disagreed on reads back as %+v (%v), want the first written", settled, err)
	}

	// Another arm's weights, another update, template, setting, code or prompt is another group.
	once, err := updatedWeights(checkpoint, training, 1)
	if err != nil {
		t.Fatal(err)
	}
	twice, _ := updatedWeights(checkpoint, training, 2)
	if start, err := updatedWeights(checkpoint, training, 0); err != nil || start != checkpoint || once == checkpoint || once == twice || !once.Valid() {
		t.Fatalf("the weights before the first update are %s (%v) and after one and two %s and %s, want the checkpoint and two identities of their own", start, err, once, twice)
	}
	for name, change := range map[string]func(*sampledGroup){
		"other weights":    func(group *sampledGroup) { group.Weights = once },
		"another template": func(group *sampledGroup) { group.Template = "raw" },
		"another setting":  func(group *sampledGroup) { group.Temperature = 1 },
		"other code":       func(group *sampledGroup) { group.Code = "inference surface 2" },
		"another prompt":   func(group *sampledGroup) { group.Prompt = []tokenizer.TokenID{1, 15} },
	} {
		other := group
		change(&other)
		if distinct, err := loadSampled(ctx, store, other); err != nil || distinct.observed == held.observed || len(distinct.values) != 0 {
			t.Errorf("the group under %s holds %d draws (%v), want none of the first group's", name, len(distinct.values), err)
		}
	}
	unnamed := group
	unnamed.Weights = artifact.ID{}
	if _, err := loadSampled(ctx, store, unnamed); err == nil {
		t.Error("a group that names no weights was drawn from")
	}

	// The cost of keeping what was sampled, beside the text an answer kept before.
	long := sampledAnswer{End: sampledOutputLimit, Text: strings.Repeat("word ", 256)}
	for index := range 256 {
		long.Completion = append(long.Completion, tokenizer.TokenID(100000+index))
		long.LogProbabilities = append(long.LogProbabilities, math.Log(0.5)/float64(index+1))
	}
	kept, err := json.Marshal(long)
	if err != nil {
		t.Fatal(err)
	}
	before, err := json.Marshal(ExactResult{Name: "prompt 12", PromptTokens: 2, GeneratedTokens: 256, Text: long.Text})
	if err != nil {
		t.Fatal(err)
	}
	each := float64(len(kept)-len(before)) / float64(len(long.Completion))
	t.Logf("a draw of %d sampled tokens is %d bytes with its ids and log probabilities and %d as text alone: %.1f bytes a token more", len(long.Completion), len(kept), len(before), each)
	if each > 32 {
		t.Errorf("keeping what was sampled costs %.1f bytes a token, more than the 32 an id and a float64 in decimal were measured to", each)
	}
}
