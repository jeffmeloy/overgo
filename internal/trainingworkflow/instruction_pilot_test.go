package trainingworkflow

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/tensorstats"
	"overgo/internal/tokenizer"
)

// pilotFixture is a store whose active catalog holds an instruction benchmark
// over the fixture model's letters, and the model. The benchmark asks for
// every combination of six kinds of rule, sixty-three of them, and for the six
// single kinds a second time, so six groups have two prompts to stand for them.
func pilotFixture(t *testing.T) (*overgodb.Store, string) {
	t.Helper()
	ctx := t.Context()
	root := t.TempDir()
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	if _, err := modelrecipe.PublishArchitectureProfileCatalog(ctx, store); err != nil {
		t.Fatal(err)
	}
	kinds := []struct {
		id     string
		kwargs map[string]any
	}{
		{"keywords:existence", map[string]any{"keywords": []string{"a"}}},
		{"keywords:forbidden_words", map[string]any{"forbidden_words": []string{"d"}}},
		{"punctuation:no_comma", map[string]any{}},
		{"startend:end_checker", map[string]any{"end_phrase": "c"}},
		{"change_case:english_lowercase", map[string]any{}},
		{"detectable_format:json_format", map[string]any{}},
	}
	var raw []byte
	asked := func(combination int) {
		var ids []string
		var kwargs []map[string]any
		for index, kind := range kinds {
			if combination>>index&1 == 1 {
				ids, kwargs = append(ids, kind.id), append(kwargs, kind.kwargs)
			}
		}
		// A prompt of the model's own letters, so it is a few tokens of its vocabulary.
		var prompt strings.Builder
		for digits := combination; digits > 0; digits /= 4 {
			prompt.WriteString(string(rune('a' + digits%4)))
		}
		line, err := json.Marshal(map[string]any{"prompt": prompt.String(), "instruction_id_list": ids, "kwargs": kwargs})
		if err != nil {
			t.Fatal(err)
		}
		raw = append(append(raw, line...), '\n')
	}
	for combination := 1; combination < 1<<len(kinds); combination++ {
		asked(combination)
	}
	for index := range kinds {
		asked(1 << index)
	}
	source := filepath.Join(root, "source.jsonl")
	if err := os.WriteFile(source, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(raw)
	imported, err := dataset.ImportBenchmark(ctx, store, source, dataset.BenchmarkImportSpec{
		Source: "pilot-fixture", Revision: "v1", SHA256: hex.EncodeToString(digest[:]), Split: "train", Format: dataset.BenchmarkFormatJSONL, Conversion: "lm-eval/ifeval/v1",
		Fields: []dataset.FieldBinding{{Source: "prompt", Target: "prompt"}, {Source: "instruction_id_list", Target: "instruction_id_list"}, {Source: "kwargs", Target: "kwargs"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	manifest := filepath.Join(root, "manifest.json")
	if err := os.WriteFile(manifest, fmt.Appendf(nil, `{"datasets":[{"name":"ifeval/default/train","dataset":%q,"path":"","spec":{}}]}`, imported.ID), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := evaluation.CatalogLocalBenchmarks(ctx, store, manifest); err != nil {
		t.Fatal(err)
	}
	return store, writeTrainingGGUF(t)
}

// TestPilotAcceptanceIsFixedFromTheStoresCases holds the instruction pilot's
// acceptance to being fixed from the store's instruction cases and readings
// of the base model, and from nothing an arm or a held-out group answered.
//
// The cases of the store's instruction family are every prompt of it; they
// are sixty-three groups, one a combination of rules, divided among the four
// parts. The base answers the prompt of each development and selection group
// twice, from its own distribution, and what it answered is in the store
// under its weights: asked for again it is read back and nothing is sampled,
// while a promotion group's prompt was never asked. The acceptance is the one
// those numbers fix: its promotion part's groups, the two comparisons, the
// protocol, the base, a basis of the development groups' draws, and arms
// differing on at most the upper bound of the share of development groups
// whose two answers differ, and the gain more. The training specification it
// names is published, with a checkpoint after one group and each doubling to
// the pass's end. Fixed again from the same store it is the same acceptance.
//
// A gain the promotion groups cannot show is refused as not enough evidence,
// with the readings it was refused on and no acceptance; and a base that is
// no GGUF, or a specification that names no draws, is a fault.
func TestPilotAcceptanceIsFixedFromTheStoresCases(t *testing.T) {
	ctx := t.Context()
	store, model := pilotFixture(t)
	const combinations, repeated = 63, 6
	pilot := InstructionPilot{
		Model: model, Host: true, Gain: 0.6, Weights: [len(evaluation.PoolParts)]uint64{2, 1, 4, 1},
		Template: rewardPlain, Window: 24, MaxTokens: 6, Seed: 11, Code: "pilot fixture code",
		Training: InstructionTraining{Draws: 4, Passes: 1, DeviceBytes: 1 << 30},
	}
	protocol := evaluation.InstructionProtocol{Scorer: evaluation.InstructionScorer, Template: rewardPlain, Thinking: pilotThinking, Sampling: rewardSampling, Stop: pilotStop, Window: 24, MaxTokens: 6}
	outcome, err := FixInstructionPilot(ctx, store, pilot)
	if err != nil || outcome.Refused != nil {
		t.Fatal(err, outcome.Refused)
	}
	acceptance, development, promotion := outcome.Acceptance, outcome.Parts[0], outcome.Parts[2]
	if outcome.Cases != combinations+repeated || outcome.Parts[0]+outcome.Parts[1]+outcome.Parts[2]+outcome.Parts[3] != combinations || slices.Contains(outcome.Parts[:], 0) {
		t.Fatalf("%d cases in parts of %v groups, want %d cases in %d groups, some in every part", outcome.Cases, outcome.Parts, combinations+repeated, combinations)
	}
	if len(outcome.Readings) != 2 || outcome.Readings[0].Part != "development" || outcome.Readings[0].Groups != development || outcome.Readings[1].Part != "selection" || outcome.Readings[1].Groups != outcome.Parts[1] {
		t.Fatalf("the base was read on %+v, want the development and selection parts of %v", outcome.Readings, outcome.Parts)
	}
	for _, read := range outcome.Readings {
		if len(read.Passed) != rewardDraws || read.Tokens == 0 || read.Differing > read.Groups || slices.Max(read.Passed) > read.Groups {
			t.Fatalf("a reading is %+v", read)
		}
	}
	_, differing := tensorstats.ExactShareBounds(uint64(outcome.Readings[0].Differing), uint64(development), acceptance.FalseGain)
	base, err := identifyModel(model)
	if err != nil {
		t.Fatal(err)
	}
	if acceptance.Groups != uint64(promotion) || acceptance.Gain != pilot.Gain || acceptance.Discordance != min(differing+pilot.Gain, 1) || acceptance.Power < 1-acceptance.MissedGain ||
		acceptance.Base != base || acceptance.Protocol != protocol || !slices.Equal(acceptance.Comparisons, pilotComparisons) || len(acceptance.Basis) != development {
		t.Fatalf("the acceptance = %+v, want %d promotion groups, the pilot's gain and comparisons, and arms differing on at most %.3f and the gain", acceptance, promotion, differing)
	}

	// What is published: the acceptance, its split, and the training specification.
	content, found, err := artifact.ReadContent(ctx, store, acceptance.Split)
	if err != nil || !found {
		t.Fatalf("the acceptance's split is published %t (%v)", found, err)
	}
	partition, err := runrecord.ParseSplitPartition(content.Data)
	if err != nil {
		t.Fatal(err)
	}
	held := map[string][]string{}
	for index, shard := range []artifact.ID{partition.Development, partition.Selection, partition.Promotion, partition.Audit} {
		membership, found, err := dataset.LoadMembership(ctx, store, shard)
		if err != nil || !found || len(membership.Records) != outcome.Parts[index] {
			t.Fatalf("part %s holds %d groups, published %t (%v); want %d", evaluation.PoolParts[index], len(membership.Records), found, err, outcome.Parts[index])
		}
		for _, record := range membership.Records {
			held[evaluation.PoolParts[index]] = append(held[evaluation.PoolParts[index]], record.ID)
		}
	}
	var training InstructionTraining
	if content, found, err = artifact.ReadContent(ctx, store, acceptance.Training); err != nil || !found || json.Unmarshal(content.Data, &training) != nil {
		t.Fatalf("the training specification is published %t (%v)", found, err)
	}
	checkpoints := []int{1}
	for checkpoints[len(checkpoints)-1] < development {
		checkpoints = append(checkpoints, min(2*checkpoints[len(checkpoints)-1], development))
	}
	if training.Draws != 4 || training.Passes != 1 || !slices.Equal(training.Checkpoints, checkpoints) || !slices.Equal(training.Arms, pilotArms) || training.Matched == "" || training.DeviceBytes != pilot.Training.DeviceBytes {
		t.Fatalf("the training specification = %+v, want the pilot's draws, passes and memory, both arms, and checkpoints %v", training, checkpoints)
	}

	// What was asked: the development and selection groups' prompts, and no other.
	prompts, _, err := evaluation.StoreInstructionPrompts(ctx, store)
	if err != nil {
		t.Fatal(err)
	}
	tokens, err := loadTrainableTokenizer(model)
	if err != nil {
		t.Fatal(err)
	}
	drawn := func(name string) (evaluation.SampledGroup, error) {
		t.Helper()
		prompt, limit, found := prompts.Asked(name)
		encoded, err := tokens.encode(prompt)
		if err != nil || !found {
			t.Fatal(name, found, err)
		}
		group := evaluation.SampledGroup{
			Group: name, Code: pilot.Code, Weights: base, Tokenizer: base, Template: rewardPlain,
			Ends: []tokenizer.TokenID{tokenizer.TokenID(tokens.end)}, Temperature: 1, Seed: pilot.Seed, MaxTokens: min(int(pilot.MaxTokens), limit),
		}
		for _, id := range encoded {
			group.Prompt = append(group.Prompt, tokenizer.TokenID(id))
		}
		_, err = evaluation.SampleGroup(ctx, store, neverSampled{}, group, rewardDraws)
		return group, err
	}
	for _, name := range slices.Concat(held["development"], held["selection"]) {
		if _, err := drawn(name); err != nil {
			t.Fatalf("the base's answers to %s are not in the store under its weights: %v", name, err)
		}
	}
	for _, name := range held["development"] {
		group, _ := drawn(name)
		if identity, err := group.Identity(); err != nil || !slices.Contains(acceptance.Basis, identity) {
			t.Fatalf("the acceptance's basis does not name the draws of %s (%v)", name, err)
		}
	}
	for _, name := range slices.Concat(held["promotion"], held["audit"]) {
		if _, err := drawn(name); err == nil {
			t.Fatalf("the prompt of %s, a held-out group, was answered before the acceptance was fixed", name)
		}
	}

	if again, err := FixInstructionPilot(ctx, store, pilot); err != nil || again.Acceptance.ID != acceptance.ID || !slices.EqualFunc(again.Readings, outcome.Readings, func(one, other PilotReading) bool {
		return one.Differing == other.Differing && slices.Equal(one.Passed, other.Passed)
	}) {
		t.Fatalf("fixed again the acceptance is %s on %+v (%v), want %s on the same readings", again.Acceptance.ID, again.Readings, err, acceptance.ID)
	}

	// A gain the groups cannot show, and what is a fault.
	small := pilot
	small.Gain = 0.1
	refused, err := FixInstructionPilot(ctx, store, small)
	if err != nil || !errors.Is(refused.Refused, evaluation.ErrInstructionEvidence) || refused.Acceptance.ID.Valid() || len(refused.Readings) != 2 || refused.Readings[0].Differing != outcome.Readings[0].Differing {
		t.Fatalf("a gain of %.2f = (%v, %v) with acceptance %s, want it refused as not enough evidence on the same readings", small.Gain, refused.Refused, err, refused.Acceptance.ID)
	}
	directory, undrawn := pilot, pilot
	directory.Model, undrawn.Training.Draws = t.TempDir(), 1
	for name, faulty := range map[string]InstructionPilot{"a base that is no GGUF": directory, "a specification of one draw": undrawn} {
		if outcome, err := FixInstructionPilot(ctx, store, faulty); err == nil {
			t.Errorf("%s fixed %+v, want a fault", name, outcome)
		}
	}
}
