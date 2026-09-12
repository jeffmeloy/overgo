package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/sequencescore"
	"overgo/internal/testutil"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"
)

const musrNormalizationEvidence = "evidence:sha256:9904c8bf2de18160031b5c1403d67ea671f6138847fb4760f3f814654f1c0226"

type musrRetainedScorer struct {
	cases  []evaluation.MultipleChoiceCase
	scores [][]sequencescore.Score
	reads  int
}

func (s *musrRetainedScorer) ScoreContinuations(_ context.Context, prompt string, candidates []string) ([]sequencescore.Score, error) {
	if s.reads >= len(s.cases) {
		return nil, fmt.Errorf("MuSR: repeated acquisition")
	}
	c := s.cases[s.reads]
	if c.Prompt != prompt || !slices.Equal(c.Candidates, candidates) {
		return nil, fmt.Errorf("MuSR: retained request differs")
	}
	scores := slices.Clone(s.scores[s.reads])
	s.reads++
	return scores, nil
}
func TestMuSRNativeNormalizationAcceptance(t *testing.T) {
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer canonical.Close()
	read := func(value string, target any) {
		t.Helper()
		id, err := artifact.ParseID(value)
		if err != nil {
			t.Fatal(err)
		}
		content, err := artifact.RequireTypedContent(t.Context(), canonical, id)
		if err != nil {
			t.Fatal(err)
		}
		if err = json.Unmarshal(content.Data, target); err != nil {
			t.Fatal(err)
		}
	}
	var evidence struct {
		Inputs  map[string]artifact.ID
		Parents []artifact.ID
	}
	read(musrNormalizationEvidence, &evidence)
	var original evaluation.MultipleChoiceSuite
	read("profile:sha256:984c2a705fe77c67197abd4733987615bc5a45fa9faf17dc147bee19324a5fb6", &original)
	const originalPlanID = "profile:sha256:b3a2c9cbbef8ad6a5193a60fd1f3eeb6d6f9db242f04a01349d9208a20047138"
	var originalPlan struct {
		ModelDefinition artifact.ID `json:"model_definition"`
		RuntimeRecipe   artifact.ID `json:"runtime_recipe"`
		CodeCommit      string      `json:"code_commit"`
		Environment     artifact.ID `json:"environment"`
		Execution       artifact.ID `json:"execution"`
	}
	read(originalPlanID, &originalPlan)
	var execution evaluation.ExecutionPolicy
	read(originalPlan.Execution.String(), &execution)
	legacy, err := evaluation.CompileMultipleChoice(original)
	if err != nil {
		t.Fatal(err)
	}
	legacyPlan, err := evaluation.BindMultipleChoice(legacy, evaluation.ExactAuthorities{ModelDefinition: originalPlan.ModelDefinition, RuntimeRecipe: originalPlan.RuntimeRecipe, CodeCommit: originalPlan.CodeCommit, Environment: originalPlan.Environment, Execution: execution})
	if err != nil || legacyPlan.Identity().String() != originalPlanID {
		t.Fatalf("original acquisition plan identity changed: %s %v", legacyPlan.Identity(), err)
	}
	var prior evaluation.GroupedChoiceReport
	read("evaluation:sha256:5247bbfb27dd415ff61d6b8385180b6ecfebf3e8f00c68574f08c025cb6aaa86", &prior)
	var replay struct {
		Cases []struct {
			Name       string
			Choices    []string
			Answer     int
			Tokens     []uint64  `json:"token_counts"`
			Characters []uint64  `json:"character_counts"`
			Sums       []float64 `json:"reconstructed_log_likelihoods"`
			Bounds     []float64 `json:"rounding_bounds"`
			Selected   int
		}
		Denominator int
		ModelLoads  int `json:"model_loads"`
	}
	read(evidence.Inputs["tmp/qwen9-musr-normalization-reuse.json"].String(), &replay)
	var native struct {
		Cases    int
		Correct  int     `json:"character_correct"`
		Accuracy float64 `json:"acc_norm"`
		Source   string  `json:"native_source_sha256"`
	}
	read(evidence.Inputs["tmp/qwen9-musr-native-normalization-check.json"].String(), &native)
	if len(original.Cases) != 756 || len(prior.Observations) != 756 || len(replay.Cases) != 756 || replay.Denominator != 756 || replay.ModelLoads != 0 || native.Cases != 756 || native.Correct != 319 || native.Source != "0884cd7ce13ba0dc83224fcb98e1f436f4c862e4b6e1d4723f7a5d4a2b6aeccb" {
		t.Fatal("retained denominator or native authority differs")
	}
	suite := evaluation.GroupedChoiceSuite{Kind: evaluation.GroupedChoiceKind, Schema: "lm-eval/leaderboard-musr/v1.0", Source: "retained-musr-normalization", Normalization: sequencescore.NormalizationCharacters, TieBreak: evaluation.TieBreakFirst}
	scorer := &musrRetainedScorer{cases: original.Cases}
	changed := 0
	for i, c := range original.Cases {
		row, old := replay.Cases[i], prior.Observations[i]
		n := len(c.Candidates)
		if row.Name != c.Name || old.Name != c.Name || row.Answer != c.Answer || old.Answer != c.Answer || len(row.Choices) != n || len(row.Tokens) != n || len(row.Characters) != n || len(row.Sums) != n || len(row.Bounds) != n || len(old.Values) != n || row.Selected < 0 || row.Selected >= n {
			t.Fatalf("case %d binding differs", i)
		}
		scores := make([]sequencescore.Score, n)
		for j := range n {
			if " "+row.Choices[j] != c.Candidates[j] || uint64(utf8.RuneCountInString(row.Choices[j])) != row.Characters[j] || row.Sums[j] != old.Values[j]*float64(row.Tokens[j]) {
				t.Fatalf("case %s candidate %d acquisition changed", c.Name, j)
			}
			scores[j] = sequencescore.Score{LogProbability: row.Sums[j], Tokens: row.Tokens[j], Characters: row.Characters[j]}
		}
		selected, err := sequencescore.Select(scores, sequencescore.NormalizationCharacters)
		if err != nil || selected.Index != row.Selected {
			t.Fatalf("case %s native selection differs: %+v %v", c.Name, selected, err)
		}
		for j := range n {
			value := selected.Values[j]
			ulp := math.Abs(math.Nextafter(old.Values[j], math.Inf(1)) - old.Values[j])
			bound := 2*ulp*float64(row.Tokens[j])/float64(row.Characters[j]) + 2*math.Abs(math.Nextafter(value, math.Inf(1))-value)
			if bound != row.Bounds[j] {
				t.Fatalf("case %s uncertainty changed", c.Name)
			}
			if j != selected.Index && selected.Values[selected.Index]-row.Bounds[selected.Index] <= value+bound {
				t.Fatalf("case %s requires another measurement", c.Name)
			}
		}
		if selected.Index != old.Selected {
			changed++
		}
		parts := strings.Split(c.Name, "/")
		if len(parts) != 4 {
			t.Fatal("MuSR group name differs")
		}
		suite.Cases = append(suite.Cases, evaluation.DemonstratedChoice{Name: c.Name, Group: parts[2], Prompt: c.Prompt, Candidates: c.Candidates, CandidateCharacters: row.Characters, Answer: c.Answer})
		scorer.scores = append(scorer.scores, scores)
	}
	if changed != 109 {
		t.Fatalf("changed selections=%d", changed)
	}
	compiled, err := evaluation.CompileGroupedChoice(suite)
	if err != nil {
		t.Fatal(err)
	}
	// The temporary report exercises production scoring; these authorities are test fixtures, not a new model run.
	authorities := evaluation.ExactAuthorities{ModelDefinition: testutil.ArtifactID(t, artifact.KindModelDefinition, "MuSR retained-score fixture"), RuntimeRecipe: testutil.ArtifactID(t, artifact.KindRecipe, "MuSR retained-score fixture"), Environment: testutil.ArtifactID(t, artifact.KindEvidence, "MuSR retained-score fixture"), CodeCommit: "0123456789abcdef0123456789abcdef01234567", Execution: evaluation.ExecutionPolicy{Lifecycle: evaluation.LifecycleResident}}
	plan, err := evaluation.BindGroupedChoice(compiled, authorities)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err = store.Commit(t.Context(), artifact.Batch{Key: "musr/fixture-authorities", Artifacts: []artifact.Descriptor{{ID: authorities.ModelDefinition}, {ID: authorities.RuntimeRecipe}, {ID: authorities.Environment}}}); err != nil {
		t.Fatal(err)
	}
	report, err := evaluation.EvaluateGroupedChoice(t.Context(), store, scorer, compiled, plan)
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Observations) != 756 || len(report.Groups) != 3 || scorer.reads != 756 || report.Accuracy != native.Accuracy {
		t.Fatalf("reanalysis differs: cases=%d reads=%d accuracy=%g", len(report.Observations), scorer.reads, report.Accuracy)
	}
	for i, o := range report.Observations {
		if o.Selected != replay.Cases[i].Selected || !slices.Equal(o.Scores, scorer.scores[i]) {
			t.Fatalf("case %s lost raw evidence", o.Name)
		}
	}
	t.Logf("MuSR: 756 retained cases, 319 correct, 109 revised selections; native scorer agrees; zero inference loads")
}
