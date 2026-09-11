package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/evaluation"
)

func TestSelectedSmokeOracleBinding(t *testing.T) {
	model, _ := artifact.IdentifyBytes(artifact.KindModel, []byte("selected"))
	other, _ := artifact.IdentifyBytes(artifact.KindModel, []byte("other"))
	recipe, _ := artifact.IdentifyBytes(artifact.KindRecipe, []byte("recipe"))
	oracle := smokeOracle{Model: model, Recipe: recipe, Domain: "text", Chat: true, Claim: "addition", Generation: &evaluation.GeneratedAnswerCase{Name: "addition", Prompt: "2+2?", MaxTokens: 16, Answers: []string{"4"}}}
	unrelated := oracle
	unrelated.Model = other
	entry := discovery.Entry{Model: model, Recipe: recipe, Present: true}
	path := filepath.Join(t.TempDir(), "oracles.json")
	for _, tc := range []struct {
		name         string
		declarations []smokeOracle
		entries      []discovery.Entry
		selection    string
		accept       bool
	}{
		{"selected", []smokeOracle{oracle, unrelated}, []discovery.Entry{entry}, model.String(), true},
		{"full missing model", []smokeOracle{oracle, unrelated}, []discovery.Entry{entry}, "", false},
		{"selected missing oracle", []smokeOracle{unrelated}, []discovery.Entry{entry}, model.String(), false},
		{"selected duplicate", []smokeOracle{oracle, oracle}, []discovery.Entry{entry}, model.String(), false},
		{"unknown selection", []smokeOracle{oracle}, []discovery.Entry{entry}, other.String(), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			data, err := json.Marshal(tc.declarations)
			if err != nil {
				t.Fatal(err)
			}
			if err = os.WriteFile(path, data, 0600); err != nil {
				t.Fatal(err)
			}
			bound, err := readSmokeOracles(path, tc.entries, tc.selection)
			if (err == nil) != tc.accept {
				t.Fatalf("accepted=%t error=%v", tc.accept, err)
			}
			if tc.accept && (len(bound) != 1 || bound[model].Recipe != recipe) {
				t.Fatalf("wrong selected binding: %+v", bound)
			}
		})
	}
}
