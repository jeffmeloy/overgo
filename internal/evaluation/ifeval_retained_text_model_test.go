//go:build modeltest

package evaluation

import (
	"encoding/json"
	"maps"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

func TestIFEvalRetainedTextAcceptance(t *testing.T) {
	t.Parallel()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	parse := func(text string) artifact.ID {
		t.Helper()
		id, err := artifact.ParseID(text)
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	read := func(id artifact.ID, target any) {
		t.Helper()
		readRetainedEvidence(t, store, id, target)
	}
	var native struct{ Profile, Selection, Scores artifact.ID }
	read(parse("evidence:sha256:78a1241f82ce179d229248564ccffb66f291d922eba864d1a261fc94d386c839"), &native)
	var scores ifevalNativeScores
	read(native.Scores, &scores)
	if scores.Selection != native.Selection || scores.Profile != native.Profile || scores.Prompts != 541 || scores.Instructions != 834 || len(scores.Rows) != scores.Prompts {
		t.Fatal("native scoring denominator or acquisition changed")
	}
	responses := retainedIFEvalResponses(t, store, native.Selection, native.Profile)
	checkRetainedIFEvalScores(t, store, scores, responses)
}

func retainedIFEvalResponses(t *testing.T, store *overgodb.Store, selection, profile artifact.ID) map[string]string {
	t.Helper()
	var acquisition struct {
		Profile, Prior artifact.ID
		Cells          []struct {
			Name        string
			Output      artifact.ID
			ReusedPrior bool `json:"reused_prior"`
		}
	}
	readRetainedEvidence(t, store, selection, &acquisition)
	if acquisition.Profile != profile || len(acquisition.Cells) != 541 {
		t.Fatal("raw acquisition denominator changed")
	}
	prior, err := RequireEvaluationEvidence(t.Context(), store, acquisition.Prior)
	if err != nil {
		t.Fatal(err)
	}
	var old InstructionRulesReport
	readRetainedEvidence(t, store, prior.Report, &old)
	retained := map[string]string{}
	for _, row := range old.Observations {
		retained[row.Name] = row.Raw
	}
	responses := map[string]string{}
	for _, cell := range acquisition.Cells {
		if _, duplicate := responses[cell.Name]; duplicate {
			t.Fatal("duplicate raw response")
		}
		if cell.ReusedPrior {
			raw, found := retained[cell.Name]
			if !found {
				t.Fatal("retained response absent")
			}
			responses[cell.Name] = raw
		} else {
			var output struct {
				Profile artifact.ID
				Result  ExactResult
			}
			readRetainedEvidence(t, store, cell.Output, &output)
			if output.Profile != profile || output.Result.Name != cell.Name {
				t.Fatal("response acquisition binding changed")
			}
			responses[cell.Name] = output.Result.Text
		}
	}

	return responses
}

func TestNativeIFEvalInputBinding(t *testing.T) {
	t.Parallel()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	id, err := artifact.ParseID("evidence:sha256:1850c0aa24d39471d3a36a1a528760c953a8c4ed17619dc24bb2b73c3e46b37d")
	if err != nil {
		t.Fatal(err)
	}
	var original nativeIFEvalInput
	readRetainedEvidence(t, store, id, &original)
	if err := checkNativeIFEvalCases(t.Context(), store, original); err != nil {
		t.Fatal(err)
	}
	const first = "ifeval/default/train/0"
	for name, mutate := range map[string]func(*nativeIFEvalInput){
		"missing case":   func(input *nativeIFEvalInput) { delete(input.Cases, first) },
		"foreign corpus": func(input *nativeIFEvalInput) { input.Dataset = id },
		"renamed case": func(input *nativeIFEvalInput) {
			input.Cases["foreign"] = input.Cases[first]
			delete(input.Cases, first)
		},
		"changed prompt": func(input *nativeIFEvalInput) { input.Cases[first]["prompt"] = json.RawMessage(`"Forged prompt."`) },
		"extra field":    func(input *nativeIFEvalInput) { input.Cases[first]["undeclared"] = json.RawMessage(`true`) },
	} {
		t.Run(name, func(t *testing.T) {
			changed := original
			changed.Cases = maps.Clone(original.Cases)
			changed.Cases[first] = maps.Clone(original.Cases[first])
			mutate(&changed)
			if err := checkNativeIFEvalCases(t.Context(), store, changed); err == nil {
				t.Fatal("changed native input accepted")
			}
		})
	}
}
