package evaluation

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/strictjson"
	"overgo/internal/testutil"
)

type ifevalNativeScores struct {
	Selection, Profile    artifact.ID
	Prompts, Instructions int
	Rows                  []struct {
		Name          string
		Strict, Loose []bool
	}
	Metrics map[string]float64
}

// The native reference's immutable 541-prompt task corpus.
const retainedIFEvalDataset = "dataset:sha256:2823ef0130090d2c7f7b8c9f6a57963f8486dfa492c4f301cf9155ca8e4d7cd4"

func TestIFEvalRetainedTextAcceptance(t *testing.T) {
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

func checkRetainedIFEvalScores(t *testing.T, store *overgodb.Store, scores ifevalNativeScores, responses map[string]string) {
	t.Helper()
	parse := func(value string) artifact.ID {
		id, err := artifact.ParseID(value)
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	if len(responses) != 541 || len(scores.Rows) != 541 || scores.Prompts != 541 || scores.Instructions != 834 {
		t.Fatal("native scoring denominator differs")
	}
	imported, found, err := dataset.ReadBenchmarkImport(t.Context(), store, parse(retainedIFEvalDataset))
	if err != nil || !found || len(imported.Records) != scores.Prompts {
		t.Fatalf("corpus denominator: found=%t err=%v", found, err)
	}
	type verdict struct{ Strict, Loose []bool }
	byName := map[string]verdict{}
	for _, row := range scores.Rows {
		if _, duplicate := byName[row.Name]; duplicate {
			t.Fatal("duplicate native verdict")
		}
		byName[row.Name] = verdict{row.Strict, row.Loose}
	}
	parameterID := parse("profile:sha256:78fff5aa9eb2bc5015aadba767afe49003cb813117422ecfbfa4fb65a19b8a4f")
	active, bound, err := store.ResolveAlias(t.Context(), benchmarkCatalogAlias)
	if err != nil || !bound {
		t.Fatal("active benchmark catalog is absent")
	}
	catalog, found, err := benchmarkCatalogCodec.Read(t.Context(), store, active)
	if err != nil || !found {
		t.Fatal("active benchmark catalog is unreadable")
	}
	boundEntries := 0
	for _, entry := range catalog.Entries {
		if entry.Dataset == imported.ID {
			if entry.Parameters != parameterID {
				t.Fatal("active dataset parameters differ from the retained reference")
			}
			boundEntries++
		}
	}
	if boundEntries != 1 {
		t.Fatal("retained parameter catalog binding is absent or ambiguous")
	}
	parameters, err := readIFEvalParameters(t.Context(), store, imported, parameterID)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]int{"language:response_language": 31, "change_case:english_capital": 25, "change_case:english_lowercase": 39, "length_constraints:number_sentences": 52, "change_case:capital_word_frequency": 25, "length_constraints:nth_paragraph_first_word": 12, "keywords:letter_frequency": 33, "keywords:existence": 39, "keywords:forbidden_words": 49, "startend:end_checker": 26, "punctuation:no_comma": 66, "startend:quotation": 41, "detectable_format:json_format": 17, "length_constraints:number_words": 52, "keywords:frequency": 42, "detectable_content:number_placeholders": 27, "detectable_format:number_bullet_lists": 31, "detectable_format:number_highlighted_sections": 48, "detectable_format:multiple_sections": 14, "length_constraints:number_paragraphs": 27, "detectable_content:postscript": 26, "detectable_format:title": 37, "detectable_format:constrained_response": 10, "combination:two_responses": 24, "combination:repeat_prompt": 41}
	counts := map[string]int{}
	instructions := 0
	for ordinal, recordID := range imported.Records {
		record, found, err := dataset.ReadBenchmarkRecord(t.Context(), store, recordID)
		if err != nil || !found {
			t.Fatalf("native record: found=%t err=%v", found, err)
		}
		fields := map[string]json.RawMessage{}
		for _, field := range record.Fields {
			fields[field.Name] = field.Value
		}
		var ids []string
		var kwargs []map[string]json.RawMessage
		if err := json.Unmarshal(fields["instruction_id_list"], &ids); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(fields["kwargs"], &kwargs); err != nil {
			t.Fatal(err)
		}
		name := fmt.Sprintf("ifeval/default/train/%d", ordinal)
		row, found := byName[name]
		raw, present := responses[name]
		if !found || !present || len(ids) != len(kwargs) || len(ids) != len(row.Strict) || len(ids) != len(row.Loose) {
			t.Fatal("native instruction or response missing")
		}
		instructions += len(ids)
		for index, id := range ids {
			if _, selected := want[id]; !selected {
				continue
			}
			var binding *ifevalParameterBinding
			if bound, present := parameters[recordID][id]; present {
				binding = &bound
			}
			rules, mapped := ifevalRuleWithParameters(id, kwargs[index], binding)
			if !mapped || len(rules) != 1 {
				t.Fatalf("native instruction %s is not mapped once", id)
			}
			rule, err := compileInstructionRule(rules[0])
			if err != nil {
				t.Fatal(err)
			}
			strict, loose, err := evaluateInstructionViews(raw, []compiledInstructionRule{rule})
			if err != nil {
				t.Fatal(err)
			}
			if strict[0] != row.Strict[index] || loose[0] != row.Loose[index] {
				t.Errorf("%s %s: strict=%t/%t loose=%t/%t", name, id, strict[0], row.Strict[index], loose[0], row.Loose[index])
			}
			counts[id]++
		}
	}
	if instructions != scores.Instructions || !maps.Equal(counts, want) {
		t.Fatalf("instruction denominator total=%d selected=%v", instructions, counts)
	}
	strictPrompts, loosePrompts, strictInstructions, looseInstructions := 0, 0, 0, 0
	for _, row := range scores.Rows {
		if allTrue(row.Strict) {
			strictPrompts++
		}
		if allTrue(row.Loose) {
			loosePrompts++
		}
		strictInstructions += countTrue(row.Strict)
		looseInstructions += countTrue(row.Loose)
	}
	metrics := map[string]float64{
		"prompt_strict":      float64(strictPrompts) / float64(scores.Prompts),
		"prompt_loose":       float64(loosePrompts) / float64(scores.Prompts),
		"instruction_strict": float64(strictInstructions) / float64(scores.Instructions),
		"instruction_loose":  float64(looseInstructions) / float64(scores.Instructions),
	}
	if !maps.Equal(scores.Metrics, metrics) {
		t.Fatalf("metrics do not match verdict counts: got=%v want=%v", scores.Metrics, metrics)
	}
	t.Logf("native parity: 541 prompts, 834 instructions, 1668 verdicts; prompt strict=%d loose=%d; instruction strict=%d loose=%d; no model acquisition", strictPrompts, loosePrompts, strictInstructions, looseInstructions)
}

func readRetainedEvidence(t *testing.T, store *overgodb.Store, id artifact.ID, target any) {
	t.Helper()
	content, err := artifact.RequireTypedContent(t.Context(), store, id)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(content.Data, target); err != nil {
		t.Fatal(err)
	}
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

func checkIFEvalNativeReference(t *testing.T, store *overgodb.Store, nativeID artifact.ID) {
	t.Helper()
	// Keep the native implementation and its operand dependencies independent
	// of this model's responses and the Go checker under evaluation.
	type nativeReference struct {
		Version      string            `json:"reference_version"`
		Sources      map[string]string `json:"sources"`
		Packages     map[string]string `json:"runtime_packages"`
		Dependencies map[string]string `json:"runtime_dependencies"`
		InputSHA256  string            `json:"native_input_sha256"`
		Input        artifact.ID       `json:"native_input"`
		LanguageSeed int               `json:"langdetect_seed"`
		RandomSeed   int               `json:"random_seed"`
	}
	var previous struct{ Scores artifact.ID }
	referenceID, err := artifact.ParseID("evidence:sha256:78a1241f82ce179d229248564ccffb66f291d922eba864d1a261fc94d386c839")
	if err != nil {
		t.Fatal(err)
	}
	readRetainedEvidence(t, store, referenceID, &previous)
	var reference, actual nativeReference
	readRetainedEvidence(t, store, previous.Scores, &reference)
	readRetainedEvidence(t, store, nativeID, &actual)
	if actual.Version != "0.4.9.1" || actual.Version != reference.Version ||
		actual.LanguageSeed != 0 || actual.RandomSeed != 0 ||
		len(actual.Sources) == 0 || len(actual.Dependencies) == 0 ||
		!maps.Equal(actual.Sources, reference.Sources) || !maps.Equal(actual.Packages, reference.Packages) ||
		!maps.Equal(actual.Dependencies, reference.Dependencies) {
		t.Fatal("pinned native judge or task inputs changed")
	}
	if actual.Input.Valid() {
		content, err := artifact.RequireTypedContent(t.Context(), store, actual.Input)
		if err != nil {
			t.Fatal(err)
		}
		contract := artifact.JSONContract(artifact.KindEvidence, "overgo/ifeval-native-input/v1")
		if err := contract.ValidateContent(content, actual.Input); err != nil || actual.InputSHA256 != actual.Input.DigestHex() {
			t.Fatal("native input content differs from its recorded hash or contract")
		}
		var input nativeIFEvalInput
		if err := strictjson.DecodeBytes(content.Data, &input); err != nil {
			t.Fatal(err)
		}
		if err := checkNativeIFEvalCases(t.Context(), store, input); err != nil {
			t.Fatal(err)
		}
	} else if actual.InputSHA256 != reference.InputSHA256 {
		t.Fatal("legacy native input identity changed without retained task inputs")
	}
}

// Retain only fields the native judge reads, bound to the frozen task corpus.
type nativeIFEvalInput struct {
	Dataset artifact.ID                           `json:"dataset"`
	Cases   map[string]map[string]json.RawMessage `json:"cases"`
}

func checkNativeIFEvalCases(ctx context.Context, reader artifact.Reader, input nativeIFEvalInput) error {
	if input.Dataset.String() != retainedIFEvalDataset {
		return errors.New("native input differs from the frozen corpus")
	}
	imported, found, err := dataset.ReadBenchmarkImport(ctx, reader, input.Dataset)
	if err != nil || !found {
		return errors.Join(err, errors.New("native input corpus is absent"))
	}
	if len(input.Cases) != len(imported.Records) {
		return errors.New("native input case denominator differs")
	}
	for ordinal, id := range imported.Records {
		record, found, err := dataset.ReadBenchmarkRecord(ctx, reader, id)
		if err != nil || !found {
			return errors.Join(err, errors.New("native input record is absent"))
		}
		name := fmt.Sprintf("ifeval/default/train/%d", ordinal)
		fields, found := input.Cases[name]
		if !found || len(fields) != len(record.Fields) {
			return fmt.Errorf("native input fields differ: %s", name)
		}
		for _, field := range record.Fields {
			value, present := fields[field.Name]
			if !present {
				return fmt.Errorf("native input field absent: %s/%s", name, field.Name)
			}
			want, err := json.Marshal(field.Value)
			if err != nil {
				return err
			}
			got, err := json.Marshal(value)
			if err != nil || !bytes.Equal(got, want) {
				return fmt.Errorf("native input value differs: %s/%s", name, field.Name)
			}
		}
	}
	return nil
}

func TestNativeIFEvalInputBinding(t *testing.T) {
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
