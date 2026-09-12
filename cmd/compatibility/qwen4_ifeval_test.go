package main

import (
	"encoding/json"
	"fmt"
	"slices"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

const qwenIFEvalProfile = "evidence:sha256:cb3c98ead452ad292b511e8d58dfb40f16f70b0c42e0cb096a0e8af960654257"
const qwenIFEvalCollector = "evidence:sha256:eb6b3d8786a2fe8e81f8b0d6bec0ccfb5946d24b0d61c0664d6461ce6a5b879b"
const qwenIFEvalContinuation = "evidence:sha256:b31245a6a126be95a7aa70bf6e2c9ec6f7f647e52edfa4569896644ce8b131ca"

type qwenIFEvalResponse struct {
	Collector, Profile artifact.ID
	Name, Prompt       string
	ShapedPrompt       string `json:"shaped_prompt"`
	Input, Output      []int32
	Result             evaluation.ExactResult
}

func checkQwenIFEvalResponse(row qwenIFEvalResponse, name, prompt string) error {
	if row.Profile.String() != qwenIFEvalProfile ||
		(row.Collector.String() != qwenIFEvalCollector && row.Collector.String() != qwenIFEvalContinuation) ||
		row.Name != name || row.Result.Name != name || row.Prompt != prompt || row.ShapedPrompt == "" ||
		len(row.Input) == 0 || len(row.Input) != row.Result.PromptTokens ||
		len(row.Output) != row.Result.GeneratedTokens || len(row.Output) > 1280 {
		return fmt.Errorf("Qwen4 IFEval response %s differs from its frozen input or producer", name)
	}
	return nil
}

func TestQwenFourIFEvalAcquisitionAcceptance(t *testing.T) {
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	parse := func(value string) artifact.ID {
		t.Helper()
		id, err := artifact.ParseID(value)
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	read := func(id artifact.ID, target any) {
		t.Helper()
		content, err := artifact.RequireTypedContent(t.Context(), store, id)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(content.Data, target); err != nil {
			t.Fatal(err)
		}
	}
	profileID := parse(qwenIFEvalProfile)
	selectionID, found, err := store.ResolveAlias(t.Context(), "validation/qwen4-ifeval/"+profileID.DigestHex()+"/acquired")
	if err != nil || !found {
		t.Fatalf("complete Qwen4 IFEval acquisition absent: found=%t err=%v", found, err)
	}
	var selection struct {
		Profile, Collector, Attempt, Continuation artifact.ID
		CumulativeBudgetNS                        int64 `json:"cumulative_budget_ns"`
		Cells                                     []ifevalCell
	}
	read(selectionID, &selection)
	if selection.Profile != profileID ||
		(selection.Collector.String() != qwenIFEvalCollector && selection.Collector.String() != qwenIFEvalContinuation) || !selection.Attempt.Valid() {
		t.Fatal("selection changed its profile, collector or acquisition attempt")
	}
	if err := checkIFEvalSelection(selection.Cells, nil); err != nil {
		t.Fatal(err)
	}
	var profile struct {
		Producer, Model, Recipe, Protocol string
		Collector, Environment            artifact.ID
		Imports                           []artifact.ID
		Prompts, Instructions             int
		MaximumTokens                     int `json:"maximum_tokens"`
	}
	read(profileID, &profile)
	if profile.Producer != "00188face63efba7c3466bb88ff00f88c8cf20be" ||
		profile.Model != "model:sha256:887f41243686e101a314013a168b6bad5df19f8bee5df3e45d942ff79455413b" ||
		profile.Recipe != "recipe:sha256:49d1ed7c41e45abcc9c10ff13ec247ab775f7054123d5507caa1f63c0a19f803" ||
		profile.Environment.String() != "evidence:sha256:828ff2a9b266308a631690d1ecd9f95235807a572a72702b6a16528473cf7633" ||
		profile.Collector.String() != qwenIFEvalCollector || profile.Prompts != 541 || profile.Instructions != 834 || profile.MaximumTokens != 1280 ||
		len(profile.Imports) != 1 || profile.Imports[0].String() != miniIFEvalDataset ||
		profile.Protocol != "one user turn, declared chat template, add-generation-prompt, thinking disabled, parse special tokens, greedy device sampling, natural EOG" {
		t.Fatal("acquisition protocol, source, environment or denominator differs")
	}
	var attempt struct{ Started, Deadline time.Time }
	read(selection.Attempt, &attempt)
	if attempt.Started.IsZero() || attempt.Deadline.Sub(attempt.Started) != 30*time.Minute {
		t.Fatal("acquisition lost its original bounded attempt")
	}
	if selection.Continuation.Valid() {
		var continuation struct{ Started, Deadline time.Time }
		read(selection.Continuation, &continuation)
		if selection.Collector.String() != qwenIFEvalContinuation ||
			!continuation.Started.Equal(attempt.Deadline) || continuation.Deadline.Sub(attempt.Started) != time.Hour ||
			selection.CumulativeBudgetNS != time.Hour.Nanoseconds() {
			t.Fatal("continuation changed its publisher or cumulative acquisition budget")
		}
	} else if selection.Collector.String() != qwenIFEvalCollector || selection.CumulativeBudgetNS != 0 {
		t.Fatal("selection names a continuation publisher without a bounded continuation")
	}
	imported, found, err := dataset.ReadBenchmarkImport(t.Context(), store, profile.Imports[0])
	if err != nil || !found || len(imported.Records) != 541 {
		t.Fatalf("native corpus absent or changed: found=%t err=%v", found, err)
	}
	cells := make(map[string]ifevalCell, len(selection.Cells))
	for _, cell := range selection.Cells {
		cells[cell.Name] = cell
	}
	instructions, capped, continued, tokens := 0, 0, 0, 0
	var inferenceWall time.Duration
	for index, id := range imported.Records {
		record, found, err := dataset.ReadBenchmarkRecord(t.Context(), store, id)
		if err != nil || !found {
			t.Fatalf("native record absent: found=%t err=%v", found, err)
		}
		var prompt string
		var ids []string
		for _, field := range record.Fields {
			switch field.Name {
			case "prompt":
				if err := json.Unmarshal(field.Value, &prompt); err != nil {
					t.Fatal(err)
				}
			case "instruction_id_list":
				if err := json.Unmarshal(field.Value, &ids); err != nil {
					t.Fatal(err)
				}
			}
		}
		if prompt == "" || len(ids) == 0 {
			t.Fatal("native prompt or instruction missing")
		}
		instructions += len(ids)
		name := fmt.Sprintf("ifeval/default/train/%d", index)
		var row qwenIFEvalResponse
		read(cells[name].Output, &row)
		if err := checkQwenIFEvalResponse(row, name, prompt); err != nil {
			t.Fatal(err)
		}
		if len(row.Output) == profile.MaximumTokens {
			capped++
		}
		if row.Collector.String() == qwenIFEvalContinuation {
			continued++
		}
		tokens += row.Result.GeneratedTokens
		inferenceWall += time.Duration(row.Result.WallNS)
		if index == 0 {
			for _, mutate := range []func(*qwenIFEvalResponse){
				func(r *qwenIFEvalResponse) { r.Profile = artifact.ID{} },
				func(r *qwenIFEvalResponse) { r.Collector = artifact.ID{} },
				func(r *qwenIFEvalResponse) { r.Prompt += " changed" },
				func(r *qwenIFEvalResponse) { r.ShapedPrompt = "" },
				func(r *qwenIFEvalResponse) { r.Result.GeneratedTokens++ },
				func(r *qwenIFEvalResponse) { r.Result.Name = "foreign" },
			} {
				corrupt := row
				mutate(&corrupt)
				if checkQwenIFEvalResponse(corrupt, name, prompt) == nil {
					t.Fatal("accepted a changed response binding")
				}
			}
		}
	}
	if instructions != 834 {
		t.Fatal("native instruction denominator changed")
	}
	if continued > 0 && !selection.Continuation.Valid() {
		t.Fatal("continued responses lack their bounded continuation")
	}
	for _, mutate := range []func([]ifevalCell) []ifevalCell{
		func(c []ifevalCell) []ifevalCell { return c[1:] },
		func(c []ifevalCell) []ifevalCell { c[1] = c[0]; return c },
		func(c []ifevalCell) []ifevalCell { c[0].ReusedPrior = true; return c },
		func(c []ifevalCell) []ifevalCell { c[0].Name = "foreign"; return c },
	} {
		if checkIFEvalSelection(mutate(slices.Clone(selection.Cells)), nil) == nil {
			t.Fatal("accepted an incomplete, duplicate, foreign or incorrectly reused selection")
		}
	}
	t.Logf("541 responses / 834 instructions; capped=%d continued=%d generated_tokens=%d inference_wall=%s; retained source and token bindings; verification model loads=0; scoring remains separate", capped, continued, tokens, inferenceWall)
}
