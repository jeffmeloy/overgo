package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

const miniIFEvalProfile = "evidence:sha256:c6ed7ddf243dcf1aa9a8f8ccffda40d2a5d31ccd3162baad7fa60ea7af400daa"
const miniIFEvalPrior = "evidence:sha256:59f9c13f5b2a01b391d0bc68090067e714b32f16b319add9d4194a8cd9bd2745"
const miniIFEvalDataset = "dataset:sha256:2823ef0130090d2c7f7b8c9f6a57963f8486dfa492c4f301cf9155ca8e4d7cd4"
const miniIFEvalPublisher = "evidence:sha256:cac0cc28fd3404c8f99a98165c2e422eef7aaf47fb6564639827eacec00d95b4"

type ifevalCell struct {
	Name        string      `json:"name"`
	Output      artifact.ID `json:"output"`
	ReusedPrior bool        `json:"reused_prior"`
}

func checkIFEvalSelection(cells []ifevalCell, retained map[string]bool) error {
	if len(cells) != 541 {
		return fmt.Errorf("IFEval: incomplete native denominator")
	}
	seen := make(map[string]bool, len(cells))
	for _, cell := range cells {
		if seen[cell.Name] || cell.ReusedPrior != retained[cell.Name] || cell.Output.Valid() == cell.ReusedPrior {
			return fmt.Errorf("IFEval: duplicate or invalid acquisition for %q", cell.Name)
		}
		seen[cell.Name] = true
	}
	for index := range 541 {
		if !seen[fmt.Sprintf("ifeval/default/train/%d", index)] {
			return fmt.Errorf("IFEval: native case %d is missing", index)
		}
	}
	return nil
}

func TestMiniCPMIFEvalAcquisitionAcceptance(t *testing.T) {
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
		value, err := artifact.RequireTypedContent(t.Context(), store, id)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(value.Data, target); err != nil {
			t.Fatal(err)
		}
	}
	profileID := parse(miniIFEvalProfile)
	selectionID, found, err := artifact.ResolveAlias(t.Context(), store, "validation/minicpm-ifeval/acquired/"+profileID.DigestHex())
	if err != nil || !found {
		t.Fatalf("complete IFEval raw acquisition absent: found=%t err=%v", found, err)
	}
	var selection struct {
		Profile artifact.ID  `json:"profile"`
		Prior   artifact.ID  `json:"prior"`
		Cells   []ifevalCell `json:"cells"`
	}
	read(selectionID, &selection)
	if selection.Profile != profileID || selection.Prior.String() != miniIFEvalPrior {
		t.Fatal("IFEval selection changed its producer or predecessor")
	}
	var profile struct {
		Producer      string        `json:"producer"`
		Model         string        `json:"model"`
		Recipe        string        `json:"recipe"`
		MaximumTokens int           `json:"maximum_tokens"`
		Imports       []artifact.ID `json:"imports"`
	}
	read(profileID, &profile)
	if profile.Producer != "f278ca230121df1d0bedf8fc58bf7881a9ebc475" || profile.Model != miniNativeModel ||
		profile.Recipe != miniNativeRecipe || profile.MaximumTokens != 1280 || len(profile.Imports) != 1 || profile.Imports[0].String() != miniIFEvalDataset {
		t.Fatal("IFEval acquisition protocol or input identities differ")
	}
	prior, err := evaluation.RequireEvaluationEvidence(t.Context(), store, parse(miniIFEvalPrior))
	if err != nil {
		t.Fatal(err)
	}
	var report evaluation.InstructionRulesReport
	read(prior.Report, &report)
	retained := make(map[string]bool)
	for _, observation := range report.Observations {
		if observation.Generated < 256 {
			retained[observation.Name] = true
		}
	}
	if len(retained) != 97 {
		t.Fatal("retained MiniCPM denominator changed")
	}
	if err := checkIFEvalSelection(selection.Cells, retained); err != nil {
		t.Fatal(err)
	}
	imported, found, err := dataset.ReadBenchmarkImport(t.Context(), store, parse(miniIFEvalDataset))
	if err != nil || !found || len(imported.Records) != 541 {
		t.Fatalf("native corpus: found=%t records=%d err=%v", found, len(imported.Records), err)
	}
	cells := make(map[string]ifevalCell, len(selection.Cells))
	for _, cell := range selection.Cells {
		cells[cell.Name] = cell
	}
	instructions, acquired := 0, 0
	for index, id := range imported.Records {
		record, found, err := dataset.ReadBenchmarkRecord(t.Context(), store, id)
		if err != nil || !found {
			t.Fatalf("native record %d: found=%t err=%v", index, found, err)
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
		instructions += len(ids)
		cell := cells[fmt.Sprintf("ifeval/default/train/%d", index)]
		if cell.ReusedPrior {
			continue
		}
		var output struct {
			Collector    artifact.ID            `json:"collector"`
			Profile      artifact.ID            `json:"profile"`
			Name         string                 `json:"name"`
			Prompt       string                 `json:"prompt"`
			ShapedPrompt string                 `json:"shaped_prompt"`
			Input        []int32                `json:"input"`
			Output       []int32                `json:"output"`
			Result       evaluation.ExactResult `json:"result"`
		}
		read(cell.Output, &output)
		if output.Collector.Valid() && output.Collector.String() != miniIFEvalPublisher ||
			output.Profile != profileID || output.Name != cell.Name || output.Result.Name != cell.Name ||
			output.Prompt != prompt || output.ShapedPrompt == "" || len(output.Input) == 0 ||
			len(output.Input) != output.Result.PromptTokens || len(output.Output) != output.Result.GeneratedTokens || len(output.Output) > 1280 {
			t.Fatalf("IFEval response %s differs from its native input or token contract", cell.Name)
		}
		acquired++
	}
	if instructions != 834 || acquired != 444 {
		t.Fatalf("IFEval instructions=%d acquired=%d, want 834/444", instructions, acquired)
	}
	for _, mutate := range []func([]ifevalCell) []ifevalCell{
		func(cells []ifevalCell) []ifevalCell { return cells[1:] },
		func(cells []ifevalCell) []ifevalCell { cells[1] = cells[0]; return cells },
		func(cells []ifevalCell) []ifevalCell {
			cells[0].ReusedPrior = !cells[0].ReusedPrior
			return cells
		},
		func(cells []ifevalCell) []ifevalCell { cells[0].Name = "not-a-native-case"; return cells },
	} {
		if err := checkIFEvalSelection(mutate(slices.Clone(selection.Cells)), retained); err == nil {
			t.Fatal("IFEval selection accepted a missing, duplicate, misclassified or foreign case")
		}
	}
	t.Log("541 native prompts / 834 instructions: 97 retained terminal responses, 444 acquired; no model execution or scorer promotion")
}

type miniIFEvalNativeRow struct {
	Name   string `json:"name"`
	Strict []bool `json:"strict"`
	Loose  []bool `json:"loose"`
}

func checkMiniIFEvalNativeScores(rows []miniIFEvalNativeRow, counts map[string]int, metrics map[string]float64) error {
	if len(rows) != 541 || len(counts) != 541 || len(metrics) != 4 {
		return fmt.Errorf("IFEval native scores: incomplete denominator")
	}
	seen := make(map[string]bool, len(rows))
	totals := make(map[string]int)
	instructions := 0
	for _, row := range rows {
		count := counts[row.Name]
		if seen[row.Name] || count <= 0 || len(row.Strict) != count || len(row.Loose) != count {
			return fmt.Errorf("IFEval native scores: duplicate, foreign or incomplete case %q", row.Name)
		}
		seen[row.Name] = true
		instructions += count
		for view, values := range map[string][]bool{"strict": row.Strict, "loose": row.Loose} {
			passed := 0
			for _, value := range values {
				if value {
					passed++
				}
			}
			totals["instruction_"+view] += passed
			if passed == len(values) {
				totals["prompt_"+view]++
			}
		}
	}
	if instructions != 834 {
		return fmt.Errorf("IFEval native scores: instruction denominator differs")
	}
	for metric, denominator := range map[string]int{"prompt_strict": 541, "prompt_loose": 541, "instruction_strict": 834, "instruction_loose": 834} {
		observed, present := metrics[metric]
		if !present || observed != float64(totals[metric])/float64(denominator) {
			return fmt.Errorf("IFEval native scores: inconsistent %s", metric)
		}
	}
	return nil
}

func TestMiniCPMIFEvalNativeScoringAcceptance(t *testing.T) {
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	read := func(id artifact.ID, target any) {
		t.Helper()
		value, err := artifact.RequireTypedContent(t.Context(), store, id)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(value.Data, target); err != nil {
			t.Fatal(err)
		}
	}
	profile, err := artifact.ParseID(miniIFEvalProfile)
	if err != nil {
		t.Fatal(err)
	}
	resolve := func(prefix string) artifact.ID {
		t.Helper()
		id, found, err := artifact.ResolveAlias(t.Context(), store, prefix+profile.DigestHex())
		if err != nil || !found {
			t.Fatalf("IFEval selection %s: found=%t err=%v", prefix, found, err)
		}
		return id
	}
	var selection struct {
		Profile   artifact.ID            `json:"profile"`
		Selection artifact.ID            `json:"selection"`
		Scores    artifact.ID            `json:"scores"`
		Inputs    map[string]artifact.ID `json:"inputs"`
	}
	read(resolve("validation/minicpm-ifeval/native/"), &selection)
	if selection.Profile != profile || selection.Selection != resolve("validation/minicpm-ifeval/acquired/") {
		t.Fatal("native scorer input differs from the complete acquisition")
	}
	source, err := artifact.RequireTypedContent(t.Context(), store, selection.Inputs["scorer_source"])
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(source.Data)) != "a009924b67be6dfc1b1dba940c20b9806b764029ef865295c9400625ba8b27e7" {
		t.Fatal("native scoring program differs from the frozen acquisition consumer")
	}
	var scores struct {
		Selection        artifact.ID           `json:"selection"`
		Profile          artifact.ID           `json:"profile"`
		ReferenceVersion string                `json:"reference_version"`
		LanguageSeed     *int                  `json:"langdetect_seed"`
		RandomSeed       *int                  `json:"random_seed"`
		Sources          map[string]string     `json:"sources"`
		Prompts          int                   `json:"prompts"`
		Instructions     int                   `json:"instructions"`
		Rows             []miniIFEvalNativeRow `json:"rows"`
		Metrics          map[string]float64    `json:"metrics"`
	}
	read(selection.Scores, &scores)
	if scores.Selection != selection.Selection || scores.Profile != profile || scores.ReferenceVersion != "0.4.9.1" ||
		scores.LanguageSeed == nil || *scores.LanguageSeed != 0 || scores.RandomSeed == nil || *scores.RandomSeed != 0 ||
		scores.Prompts != 541 || scores.Instructions != 834 ||
		scores.Sources["instructions.py"] != "1556285b56cb81a7a1ada79327cd8797681aed004160c7f87b3a94f3e10bae32" ||
		scores.Sources["instructions_registry.py"] != "9db1c062cdb70a91420d3789a588235a1064c2b9a1a66e09e5fb9df9e7584a6c" ||
		scores.Sources["instructions_util.py"] != "e8c4d9187bac1482d93941fb46469609c3ae78d896195bfcb300a0707d492567" ||
		scores.Sources["utils.py"] != "1ab8f14808c826f93f2364883487ed63cf4267980bf4761fda8053899c013632" {
		t.Fatal("native scorer protocol, producer or denominator differs")
	}
	datasetID, err := artifact.ParseID(miniIFEvalDataset)
	if err != nil {
		t.Fatal(err)
	}
	imported, found, err := dataset.ReadBenchmarkImport(t.Context(), store, datasetID)
	if err != nil || !found {
		t.Fatalf("native input: found=%t err=%v", found, err)
	}
	counts := make(map[string]int)
	for index, id := range imported.Records {
		record, found, err := dataset.ReadBenchmarkRecord(t.Context(), store, id)
		if err != nil || !found {
			t.Fatalf("native case %d: found=%t err=%v", index, found, err)
		}
		for _, field := range record.Fields {
			if field.Name == "instruction_id_list" {
				var instructions []string
				if err := json.Unmarshal(field.Value, &instructions); err != nil {
					t.Fatal(err)
				}
				counts[fmt.Sprintf("ifeval/default/train/%d", index)] = len(instructions)
			}
		}
	}
	if err := checkMiniIFEvalNativeScores(scores.Rows, counts, scores.Metrics); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func([]miniIFEvalNativeRow) []miniIFEvalNativeRow{
		func(rows []miniIFEvalNativeRow) []miniIFEvalNativeRow { return rows[1:] },
		func(rows []miniIFEvalNativeRow) []miniIFEvalNativeRow { rows[1] = rows[0]; return rows },
		func(rows []miniIFEvalNativeRow) []miniIFEvalNativeRow { rows[0].Strict = nil; return rows },
	} {
		if err := checkMiniIFEvalNativeScores(mutate(slices.Clone(scores.Rows)), counts, scores.Metrics); err == nil {
			t.Fatal("native scorer accepted omitted, duplicate or incomplete results")
		}
	}
	scores.Metrics["prompt_strict"]++
	if err := checkMiniIFEvalNativeScores(scores.Rows, counts, scores.Metrics); err == nil {
		t.Fatal("native scorer accepted inconsistent reported totals")
	}
	t.Log("541 prompts and 834 instruction outcomes scored by pinned native lm_eval; no Go scorer parity or quality-floor promotion")
}
