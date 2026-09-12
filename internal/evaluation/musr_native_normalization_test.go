package evaluation

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/sequencescore"
	"slices"
	"testing"
)

//go:embed testdata/musr_normalization_native.json
var musrNormalizationOracle []byte

func TestMuSRNativeNormalization(t *testing.T) {
	var oracle struct {
		Source   string `json:"native_source_sha256"`
		Fixtures []struct {
			Name             string
			Choices          []string
			LogProbabilities []float64 `json:"log_probabilities"`
			TokenCounts      []uint64  `json:"token_counts"`
			CharacterCounts  []uint64  `json:"character_counts"`
			Values           []float64
			Selected         int
			Tied             bool
		}
	}
	if err := json.Unmarshal(musrNormalizationOracle, &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.Source != "0884cd7ce13ba0dc83224fcb98e1f436f4c862e4b6e1d4723f7a5d4a2b6aeccb" || len(oracle.Fixtures) != 3 {
		t.Fatal("native scorer fixture differs")
	}
	for _, fixture := range oracle.Fixtures {
		t.Run(fixture.Name, func(t *testing.T) {
			candidates := make([]string, len(fixture.Choices))
			scores := make([]sequencescore.Score, len(candidates))
			for i := range candidates {
				candidates[i] = " " + fixture.Choices[i]
				scores[i] = sequencescore.Score{LogProbability: fixture.LogProbabilities[i], Tokens: fixture.TokenCounts[i]}
			}
			source := GroupedChoiceSuite{Kind: GroupedChoiceKind, Schema: "native-musr-normalization/v1", Source: "fixture", Normalization: sequencescore.NormalizationCharacters, TieBreak: TieBreakFirst, Cases: []DemonstratedChoice{{Name: fixture.Name, Group: "native", Prompt: "Question\nAnswer:", Candidates: candidates, CandidateCharacters: slices.Clone(fixture.CharacterCounts), Answer: fixture.Selected}}}
			compiled, err := CompileGroupedChoice(source)
			if err != nil {
				t.Fatal(err)
			}
			source.Cases[0].CandidateCharacters[0]++
			if !slices.Equal(compiled.choice.suite.Cases[0].CandidateCharacters, fixture.CharacterCounts) {
				t.Fatal("compiled character counts alias the caller")
			}
			plan, err := BindGroupedChoice(compiled, ExactAuthorities{ModelDefinition: planID(t, artifact.KindModelDefinition, "model"), RuntimeRecipe: planID(t, artifact.KindRecipe, "recipe"), CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "environment"), Execution: ExecutionPolicy{Lifecycle: LifecycleResident}})
			if err != nil {
				t.Fatal(err)
			}
			store, err := overgodb.Open(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			publishPlanFixtureAuthorities(t, store, plan)
			report, err := EvaluateGroupedChoice(t.Context(), store, fixedContinuationScorer{scores: scores}, compiled, plan)
			if err != nil {
				t.Fatal(err)
			}
			o := report.Observations[0]
			if o.Selected != fixture.Selected || o.Tied != fixture.Tied || !slices.Equal(o.Values, fixture.Values) || report.Accuracy != 1 || len(report.Groups) != 1 || report.Groups[0].Correct != 1 {
				t.Fatalf("native scoring differs: %+v", report)
			}
			if len(o.Scores) != len(scores) {
				t.Fatal("raw likelihoods not retained")
			}
			for i, s := range o.Scores {
				if s.LogProbability != scores[i].LogProbability || s.Tokens != scores[i].Tokens || s.Characters != fixture.CharacterCounts[i] {
					t.Fatal("raw acquisition or denominator changed")
				}
			}
			scores[0].LogProbability = 0
			if o.Scores[0].LogProbability == 0 {
				t.Fatal("stored raw scores alias the scorer")
			}
			selected, err := sequencescore.Select(o.Scores, sequencescore.NormalizationCharacters)
			if err != nil || selected.Index != o.Selected || !slices.Equal(selected.Values, o.Values) {
				t.Fatalf("retained scores cannot replay: %+v %v", selected, err)
			}
		})
	}
	t.Run("assembler", func(t *testing.T) {
		value, _, err := assembleMuSRSuite([]storeCase{{entry: "musr/default/fixture", subset: "fixture", fields: rawFields(t, map[string]any{"narrative": "Story", "question": "Who?", "choices": "['é', 'xx']", "answer_index": 0})}})
		if err != nil {
			t.Fatal(err)
		}
		suite := value.(GroupedChoiceSuite)
		if suite.Normalization != sequencescore.NormalizationCharacters || suite.TieBreak != TieBreakFirst || !slices.Equal(suite.Cases[0].CandidateCharacters, []uint64{1, 2}) || !slices.Equal(suite.Cases[0].Candidates, []string{" é", " xx"}) {
			t.Fatalf("native source characters/delimiter: %+v", suite)
		}
	})
	t.Run("invalid denominators", func(t *testing.T) {
		for _, counts := range [][]uint64{nil, {1}, {1, 0}, {1, 2, 3}} {
			_, err := CompileMultipleChoice(MultipleChoiceSuite{Kind: MultipleChoiceKind, Schema: "fixture/v1", Source: "fixture", Normalization: sequencescore.NormalizationCharacters, Aggregation: AggregationAccuracy, Cases: []MultipleChoiceCase{{Name: "bad", Prompt: "Q", Candidates: []string{" a", " bb"}, CandidateCharacters: counts, Answer: 0}}})
			if err == nil {
				t.Fatalf("accepted denominators %v", counts)
			}
		}
		if _, err := sequencescore.Select([]sequencescore.Score{{LogProbability: -1, Tokens: 1}, {LogProbability: -2, Tokens: 1}}, sequencescore.NormalizationCharacters); err == nil {
			t.Fatal("sequence scorer accepted absent characters")
		}
	})
	t.Run("legacy wire", func(t *testing.T) {
		fixtures := []struct {
			data  string
			value any
		}{{`{"log_probability":-2,"tokens":1}`, &sequencescore.Score{}}, {`{"name":"legacy","prompt":"Q","candidates":[" a"," bb"],"answer":0}`, &MultipleChoiceCase{}}, {`{"name":"legacy","values":[-2,-3],"selected":0,"answer":0,"tied":false}`, &ChoiceObservation{}}}
		for _, f := range fixtures {
			if err := json.Unmarshal([]byte(f.data), f.value); err != nil {
				t.Fatal(err)
			}
			data, err := json.Marshal(f.value)
			if err != nil || !bytes.Equal(data, []byte(f.data)) {
				t.Fatalf("legacy wire changed: %s %v", data, err)
			}
		}
		choice, err := sequencescore.Select([]sequencescore.Score{{LogProbability: -2, Tokens: 1, Characters: 1}, {LogProbability: -4, Tokens: 1, Characters: 10}}, sequencescore.NormalizationMean)
		if err != nil || choice.Index != 0 {
			t.Fatalf("legacy token mean changed: %+v %v", choice, err)
		}
	})
	t.Run("distinct scoring authority", func(t *testing.T) {
		suite := GroupedChoiceSuite{Kind: GroupedChoiceKind, Schema: "fixture/v1", Source: "same-acquisition", Normalization: sequencescore.NormalizationMean, Cases: []DemonstratedChoice{{Name: "same-case", Group: "same-group", Prompt: "Q", Candidates: []string{" a", " bbbbbbbbbb"}, Answer: 1}}}
		legacy, err := CompileGroupedChoice(suite)
		if err != nil {
			t.Fatal(err)
		}
		suite.Normalization = sequencescore.NormalizationCharacters
		suite.TieBreak = TieBreakFirst
		suite.Cases[0].CandidateCharacters = []uint64{1, 10}
		native, err := CompileGroupedChoice(suite)
		if err != nil {
			t.Fatal(err)
		}
		authorities := ExactAuthorities{ModelDefinition: planID(t, artifact.KindModelDefinition, "model"), RuntimeRecipe: planID(t, artifact.KindRecipe, "recipe"), CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "environment"), Execution: ExecutionPolicy{Lifecycle: LifecycleResident}}
		oldPlan, err := BindGroupedChoice(legacy, authorities)
		if err != nil {
			t.Fatal(err)
		}
		newPlan, err := BindGroupedChoice(native, authorities)
		if err != nil {
			t.Fatal(err)
		}
		suite.TieBreak = ""
		noTie, err := CompileGroupedChoice(suite)
		if err != nil {
			t.Fatal(err)
		}
		noTiePlan, err := BindGroupedChoice(noTie, authorities)
		if err != nil {
			t.Fatal(err)
		}
		if newPlan.body.Scorer == noTiePlan.body.Scorer || newPlan.identity == noTiePlan.identity {
			t.Fatal("tie policy did not change scorer authority")
		}
		tied := ChoiceObservation{Selected: 0, Answer: 0, Tied: true}
		if choiceCorrect(tied, "") || !choiceCorrect(tied, TieBreakFirst) {
			t.Fatal("tie acceptance is not controlled by the declared policy")
		}
		suite.TieBreak = "unrecognized"
		if _, err := CompileGroupedChoice(suite); err == nil {
			t.Fatal("unknown tie policy accepted")
		}
		if legacy.choice.identity == native.choice.identity || oldPlan.identity == newPlan.identity {
			t.Fatal("changed scoring reused prior authority")
		}
		store, err := overgodb.Open(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		if _, err := EvaluateGroupedChoice(t.Context(), store, fixedContinuationScorer{}, native, oldPlan); err == nil {
			t.Fatal("native scoring accepted a legacy plan")
		}
	})
}
