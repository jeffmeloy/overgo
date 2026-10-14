package evaluation

import (
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/strictjson"
)

var ifevalParameterContract = artifact.DocumentContract{
	Kind: artifact.KindProfile, MediaType: "application/json", Schema: "overgo/ifeval-parameters/v1",
}

var ifevalReferenceContract = artifact.DocumentContract{
	Kind: artifact.KindEvidence, MediaType: "application/json", Schema: "overgo/ifeval-resolved-reference/v1",
}

//go:embed testdata/ifeval_parameters.json
var ifevalParameterFile []byte

// ifevalParameterFileProfile is the identity of the profile ifevalParameterFile holds.
//   - source = the profile the store's active catalog binds to ifeval/default/train
//   - scope = the native operands of the IFEval letter-frequency instructions in that one import
//   - reconsider <=> the native operands are captured again
const ifevalParameterFileProfile = "profile:sha256:78fff5aa9eb2bc5015aadba767afe49003cb813117422ecfbfa4fb65a19b8a4f"

// ifevalParameterRecords holds a parameter profile and the reference it names, each as its stored bytes.
type ifevalParameterRecords struct {
	Profile   json.RawMessage `json:"profile"`
	Reference json.RawMessage `json:"reference"`
}

// Parameters bind native constructor results to existing immutable records.
// Both views are explicit; neither compilation nor restart draws new operands.
type ifevalParameters struct {
	Dataset   artifact.ID              `json:"dataset"`
	Reference artifact.ID              `json:"reference"`
	Bindings  []ifevalParameterBinding `json:"bindings"`
}

type ifevalParameterBinding struct {
	Record      artifact.ID                `json:"record"`
	Instruction string                     `json:"instruction"`
	Strict      map[string]json.RawMessage `json:"strict"`
	Loose       map[string]json.RawMessage `json:"loose"`
}

// publishIFEvalParameterFile publishes the file's profile and its reference for the import of the dataset the profile names, and returns the parameters the catalog binds.
//   - imported is another dataset, or declared names another profile => declared returns and nothing is written
//   - the file's profile != ifevalParameterFileProfile => refused, because the live suite binds that identity
//   - the file's reference != the reference the profile names => refused
func publishIFEvalParameterFile(ctx context.Context, repository artifact.Repository, file []byte, imported dataset.BenchmarkImport, declared artifact.ID) (artifact.ID, error) {
	var records ifevalParameterRecords
	if err := strictjson.DecodeBytes(file, &records); err != nil {
		return artifact.ID{}, err
	}
	profile, err := ifevalParameterContract.ContentBytes(records.Profile)
	if err != nil {
		return artifact.ID{}, err
	}
	if profile.Descriptor.ID.String() != ifevalParameterFileProfile {
		return artifact.ID{}, fmt.Errorf("evaluation: IFEval parameter file holds profile %s, want %s", profile.Descriptor.ID, ifevalParameterFileProfile)
	}
	var parameters ifevalParameters
	if err := strictjson.DecodeBytes(profile.Data, &parameters); err != nil {
		return artifact.ID{}, err
	}
	if parameters.Dataset != imported.ID || (declared.Kind() != artifact.KindInvalid && declared != profile.Descriptor.ID) {
		return declared, nil
	}
	reference, err := ifevalReferenceContract.ContentBytes(records.Reference)
	if err != nil {
		return artifact.ID{}, err
	}
	if reference.Descriptor.ID != parameters.Reference {
		return artifact.ID{}, fmt.Errorf("evaluation: IFEval parameter file holds reference %s, want %s the profile names", reference.Descriptor.ID, parameters.Reference)
	}
	batch, err := artifact.NewDocumentBatch(
		"evaluation/ifeval-parameters/"+profile.Descriptor.ID.String(), []artifact.Content{reference, profile},
		artifact.DependencyLineage(profile.Descriptor.ID, reference.Descriptor.ID), nil,
	)
	if err != nil {
		return artifact.ID{}, err
	}
	if _, err := artifact.Publish(ctx, repository, batch); err != nil {
		return artifact.ID{}, err
	}
	return profile.Descriptor.ID, nil
}

func readIFEvalParameters(ctx context.Context, reader artifact.Reader, imported dataset.BenchmarkImport, id artifact.ID) (map[artifact.ID]map[string]ifevalParameterBinding, error) {
	content, err := artifact.RequireTypedContent(ctx, reader, id)
	if err != nil {
		return nil, err
	}
	if err := ifevalParameterContract.ValidateContent(content, id); err != nil {
		return nil, err
	}
	var profile ifevalParameters
	if err := strictjson.DecodeBytes(content.Data, &profile); err != nil {
		return nil, err
	}
	family, recognized := familyForConversion(imported.Spec.Conversion)
	if !recognized || family.family != "ifeval" || profile.Dataset != imported.ID || profile.Reference.Kind() != artifact.KindEvidence || len(profile.Bindings) == 0 {
		return nil, errors.New("evaluation: incompatible IFEval parameter profile")
	}
	if _, err := artifact.RequireTypedContent(ctx, reader, profile.Reference); err != nil {
		return nil, err
	}
	result := map[artifact.ID]map[string]ifevalParameterBinding{}
	for _, binding := range profile.Bindings {
		if !slices.Contains(imported.Records, binding.Record) {
			return nil, errors.New("evaluation: IFEval parameters reference an unrelated record")
		}
		if _, duplicate := result[binding.Record][binding.Instruction]; duplicate {
			return nil, errors.New("evaluation: duplicate IFEval parameter binding")
		}
		record, found, err := dataset.ReadBenchmarkRecord(ctx, reader, binding.Record)
		if err != nil || !found {
			return nil, errors.Join(err, errors.New("evaluation: IFEval parameter record is absent"))
		}
		var ids []string
		for _, field := range record.Fields {
			if field.Name == "instruction_id_list" {
				if err := json.Unmarshal(field.Value, &ids); err != nil {
					return nil, err
				}
			}
		}
		matches := 0
		for _, name := range ids {
			if name == binding.Instruction {
				matches++
			}
		}
		if matches != 1 {
			return nil, errors.New("evaluation: IFEval parameter instruction is absent or ambiguous")
		}
		rules, ok := ifevalRuleWithParameters(binding.Instruction, nil, &binding)
		if !ok {
			return nil, errors.New("evaluation: IFEval parameter views are incomplete or unsupported")
		}
		if _, err := compileInstructionRule(rules[0]); err != nil {
			return nil, err
		}
		if result[binding.Record] == nil {
			result[binding.Record] = map[string]ifevalParameterBinding{}
		}
		result[binding.Record][binding.Instruction] = binding
	}
	return result, nil
}

func ifevalRuleWithParameters(id string, kwargs map[string]json.RawMessage, binding *ifevalParameterBinding) ([]instructionRule, bool) {
	if binding == nil {
		return ifevalRuleFor(id, kwargs)
	}
	if binding.Instruction != id || len(binding.Strict) == 0 || len(binding.Loose) == 0 {
		return nil, false
	}
	// The captured random constructor gap is the native letter checker.
	if id != "keywords:letter_frequency" {
		return nil, false
	}
	keys := []string{"letter", "let_frequency", "let_relation"}
	for _, view := range []map[string]json.RawMessage{binding.Strict, binding.Loose} {
		if len(view) != len(keys) {
			return nil, false
		}
		for _, key := range keys {
			if _, present := view[key]; !present {
				return nil, false
			}
		}
	}
	strict, strictOK := ifevalRuleFor(id, binding.Strict)
	loose, looseOK := ifevalRuleFor(id, binding.Loose)
	if !strictOK || !looseOK || len(strict) != 1 || len(loose) != 1 {
		return nil, false
	}
	strict[0].Loose = &loose[0]
	return strict, true
}
