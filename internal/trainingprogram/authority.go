// Package trainingprogram compiles immutable model-construction and training
// authority. Executors consume the sealed result; they do not infer policy.
package trainingprogram

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/optimizer"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
)

type RNGStreamSpec struct {
	Name      string
	Algorithm artifact.ID
	Seed      uint64
}

type ScratchSpec struct {
	Recipe             artifact.ID
	Dataset            artifact.ID
	Split              artifact.ID
	DerivationProfile  artifact.ID
	TopologyProfile    artifact.ID
	Tokenizer          artifact.ID
	ParameterManifest  artifact.ID
	InitializerProfile artifact.ID
	InitializedModel   artifact.ID
	RNGStreams         []RNGStreamSpec
}

type ScratchConstruction struct {
	id                 artifact.ID
	recipe             artifact.ID
	dataset            artifact.ID
	split              artifact.ID
	derivationProfile  artifact.ID
	topologyProfile    artifact.ID
	tokenizer          artifact.ID
	parameterManifest  artifact.ID
	initializerProfile artifact.ID
	initializedModel   artifact.ID
	rngStreams         []RNGStreamSpec
}

func CompileScratchConstruction(spec ScratchSpec) (ScratchConstruction, error) {
	if spec.Recipe.Kind() != artifact.KindRecipe || spec.Dataset.Kind() != artifact.KindDataset ||
		spec.Split.Kind() != artifact.KindDatasetShard || spec.DerivationProfile.Kind() != artifact.KindProfile ||
		spec.TopologyProfile.Kind() != artifact.KindProfile || spec.Tokenizer.Kind() != artifact.KindTokenizer ||
		spec.ParameterManifest.Kind() != artifact.KindTensorInventory || spec.InitializerProfile.Kind() != artifact.KindProfile ||
		spec.InitializedModel.Kind() != artifact.KindModel {
		return ScratchConstruction{}, errors.New("training program: invalid scratch construction identities")
	}
	streams, err := compileRNGStreams(spec.RNGStreams)
	if err != nil {
		return ScratchConstruction{}, err
	}
	body := struct {
		Recipe             artifact.ID     `json:"recipe"`
		Dataset            artifact.ID     `json:"dataset"`
		Split              artifact.ID     `json:"split"`
		DerivationProfile  artifact.ID     `json:"derivation_profile"`
		TopologyProfile    artifact.ID     `json:"topology_profile"`
		Tokenizer          artifact.ID     `json:"tokenizer"`
		ParameterManifest  artifact.ID     `json:"parameter_manifest"`
		InitializerProfile artifact.ID     `json:"initializer_profile"`
		InitializedModel   artifact.ID     `json:"initialized_model"`
		RNGStreams         []RNGStreamSpec `json:"rng_streams"`
	}{
		Recipe: spec.Recipe, Dataset: spec.Dataset, Split: spec.Split,
		DerivationProfile: spec.DerivationProfile, TopologyProfile: spec.TopologyProfile,
		Tokenizer: spec.Tokenizer, ParameterManifest: spec.ParameterManifest,
		InitializerProfile: spec.InitializerProfile, InitializedModel: spec.InitializedModel,
		RNGStreams: streams,
	}
	id, err := artifact.JSONID(artifact.KindRecipe, body)
	if err != nil {
		return ScratchConstruction{}, err
	}
	return ScratchConstruction{
		id: id, recipe: spec.Recipe, dataset: spec.Dataset, split: spec.Split,
		derivationProfile: spec.DerivationProfile, topologyProfile: spec.TopologyProfile,
		tokenizer: spec.Tokenizer, parameterManifest: spec.ParameterManifest,
		initializerProfile: spec.InitializerProfile, initializedModel: spec.InitializedModel,
		rngStreams: streams,
	}, nil
}

func (c ScratchConstruction) ID() artifact.ID                { return c.id }
func (c ScratchConstruction) InitializedModel() artifact.ID  { return c.initializedModel }
func (c ScratchConstruction) Dataset() artifact.ID           { return c.dataset }
func (c ScratchConstruction) Split() artifact.ID             { return c.split }
func (c ScratchConstruction) DerivationProfile() artifact.ID { return c.derivationProfile }
func (c ScratchConstruction) RNGStreams() []RNGStreamSpec    { return slices.Clone(c.rngStreams) }

type OperatorPhase string

const (
	PhaseBatch    OperatorPhase = "batch"
	PhaseForward  OperatorPhase = "forward"
	PhaseLoss     OperatorPhase = "loss"
	PhaseBackward OperatorPhase = "backward"
	PhaseOptimize OperatorPhase = "optimize"
	PhaseEvaluate OperatorPhase = "evaluate"
)

type OperatorSpec struct {
	ID    string
	Phase OperatorPhase
}

type ParameterSpec struct {
	Name      string
	Rows      int
	Cols      int
	Trainable bool
}

type ProgramSpec struct {
	Objective  ObjectiveKind
	Operators  []OperatorSpec
	Parameters []ParameterSpec
	Optimizer  optimizer.Plan
	Preference *PreferencePolicy
}

// PreferencePolicy seals reference and objective scale facts.
type PreferencePolicy struct {
	Reference artifact.ID
	Scale     float64
}

func (policy PreferencePolicy) Validate() error {
	if policy.Reference.Kind() != artifact.KindModel || policy.Scale <= 0 ||
		math.IsNaN(policy.Scale) || math.IsInf(policy.Scale, 0) {
		return errors.New("training program: DPO requires reference model and finite positive scale")
	}
	return nil
}

type TrainingProgram struct {
	id          artifact.ID
	objective   ObjectiveKind
	operators   []OperatorSpec
	parameters  []ParameterSpec
	optimizerID string
	preference  *PreferencePolicy
}

func CompileTrainingProgram(spec ProgramSpec) (TrainingProgram, error) {
	if !validObjectiveKind(spec.Objective) {
		return TrainingProgram{}, errors.New("training program: invalid objective")
	}
	operators, err := compileOperators(spec.Operators)
	if err != nil {
		return TrainingProgram{}, err
	}
	parameters, err := compileParameters(spec.Parameters, spec.Optimizer)
	if err != nil {
		return TrainingProgram{}, err
	}
	preference, err := compilePreferencePolicy(spec.Objective, spec.Preference)
	if err != nil {
		return TrainingProgram{}, err
	}
	body := struct {
		Objective   ObjectiveKind     `json:"objective"`
		Operators   []OperatorSpec    `json:"operators"`
		Parameters  []ParameterSpec   `json:"parameters"`
		OptimizerID string            `json:"optimizer_id"`
		Preference  *PreferencePolicy `json:"preference,omitempty"`
	}{Objective: spec.Objective, Operators: operators, Parameters: parameters, OptimizerID: spec.Optimizer.Identity(), Preference: preference}
	id, err := artifact.JSONID(artifact.KindRecipe, body)
	if err != nil {
		return TrainingProgram{}, err
	}
	return TrainingProgram{id: id, objective: spec.Objective, operators: operators, parameters: parameters, optimizerID: spec.Optimizer.Identity(), preference: preference}, nil
}

func (p TrainingProgram) ID() artifact.ID             { return p.id }
func (p TrainingProgram) Objective() ObjectiveKind    { return p.objective }
func (p TrainingProgram) Operators() []OperatorSpec   { return slices.Clone(p.operators) }
func (p TrainingProgram) Parameters() []ParameterSpec { return slices.Clone(p.parameters) }
func (p TrainingProgram) OptimizerIdentity() string   { return p.optimizerID }
func (p TrainingProgram) Preference() (PreferencePolicy, bool) {
	if p.preference == nil {
		return PreferencePolicy{}, false
	}
	return *p.preference, true
}

func compilePreferencePolicy(objective ObjectiveKind, policy *PreferencePolicy) (*PreferencePolicy, error) {
	if objective != ObjectiveDPO {
		if policy != nil {
			return nil, errors.New("training program: preference policy requires DPO")
		}
		return nil, nil
	}
	if policy == nil {
		return nil, errors.New("training program: DPO requires preference policy")
	}
	if err := policy.Validate(); err != nil {
		return nil, err
	}
	copy := *policy
	return &copy, nil
}

type InitialStateSpec struct {
	Model      artifact.ID
	Checkpoint artifact.ID
	Scratch    *ScratchConstruction
}

type PolicySpec struct {
	Objective  artifact.ID
	Precision  artifact.ID
	Placement  artifact.ID
	Memory     artifact.ID
	Checkpoint artifact.ID
	Evaluation artifact.ID
	Promotion  artifact.ID
}

var policyDependencyRoles = []recipe.DependencyRole{
	recipe.DependencyObjective, recipe.DependencyPrecision, recipe.DependencyPlacement,
	recipe.DependencyMemory, recipe.DependencyCheckpointPolicy, recipe.DependencyEvaluation,
	recipe.DependencyPromotion,
}

func PoliciesFromRecipe(definition recipe.Definition) (PolicySpec, error) {
	ids := make([]artifact.ID, len(policyDependencyRoles))
	for index, role := range policyDependencyRoles {
		id, ok := definition.Dependency(role, 0)
		if !ok {
			return PolicySpec{}, fmt.Errorf("training program: recipe dependency %q absent", role)
		}
		ids[index] = id
	}
	return PolicySpec{
		Objective: ids[0], Precision: ids[1], Placement: ids[2], Memory: ids[3],
		Checkpoint: ids[4], Evaluation: ids[5], Promotion: ids[6],
	}, nil
}

type RunSpec struct {
	Recipe     artifact.ID
	Initial    InitialStateSpec
	Dataset    artifact.ID
	Split      artifact.ID
	Signature  recipecontract.ModalitySignature
	Processors []artifact.ID
	Projectors []artifact.ID
	Codecs     []artifact.ID
	Policies   PolicySpec
	Program    TrainingProgram
}

type InitialStateMode string

const (
	InitialPretrained InitialStateMode = "pretrained"
	InitialResume     InitialStateMode = "resume"
	InitialScratch    InitialStateMode = "scratch"
)

type TrainingRunPlan struct {
	id         artifact.ID
	recipe     artifact.ID
	mode       InitialStateMode
	model      artifact.ID
	checkpoint artifact.ID
	scratch    ScratchConstruction
	dataset    artifact.ID
	split      artifact.ID
	signature  recipecontract.ModalitySignature
	processors []artifact.ID
	projectors []artifact.ID
	codecs     []artifact.ID
	policies   PolicySpec
	program    TrainingProgram
}

func CompileTrainingRunPlan(spec RunSpec) (TrainingRunPlan, error) {
	if spec.Recipe.Kind() != artifact.KindRecipe || spec.Dataset.Kind() != artifact.KindDataset ||
		spec.Split.Kind() != artifact.KindDatasetShard || spec.Program.ID().Kind() != artifact.KindRecipe {
		return TrainingRunPlan{}, errors.New("training program: invalid run identities")
	}
	mode, model, checkpoint, scratch, err := compileInitialState(spec.Initial)
	if err != nil {
		return TrainingRunPlan{}, err
	}
	if mode == InitialScratch && (scratch.Dataset() != spec.Dataset || scratch.Split() != spec.Split) {
		return TrainingRunPlan{}, errors.New("training program: scratch construction dataset/split differs from run")
	}
	if err := spec.Signature.Validate(); err != nil {
		return TrainingRunPlan{}, fmt.Errorf("training program: modality signature: %w", err)
	}
	processors, err := compileIDs("processor", spec.Processors, artifact.KindProfile)
	if err != nil {
		return TrainingRunPlan{}, err
	}
	projectors, err := compileIDs("projector", spec.Projectors, artifact.KindProjector)
	if err != nil {
		return TrainingRunPlan{}, err
	}
	codecs, err := compileIDs("codec", spec.Codecs, artifact.KindProfile)
	if err != nil {
		return TrainingRunPlan{}, err
	}
	if err := validatePolicies(spec.Policies); err != nil {
		return TrainingRunPlan{}, err
	}
	signature := spec.Signature.Clone()
	body := struct {
		Recipe     artifact.ID                      `json:"recipe"`
		Mode       InitialStateMode                 `json:"initial_mode"`
		Model      string                           `json:"model,omitempty"`
		Checkpoint string                           `json:"checkpoint,omitempty"`
		Scratch    string                           `json:"scratch,omitempty"`
		Dataset    artifact.ID                      `json:"dataset"`
		Split      artifact.ID                      `json:"split"`
		Signature  recipecontract.ModalitySignature `json:"signature"`
		Processors []artifact.ID                    `json:"processors,omitempty"`
		Projectors []artifact.ID                    `json:"projectors,omitempty"`
		Codecs     []artifact.ID                    `json:"codecs,omitempty"`
		Policies   PolicySpec                       `json:"policies"`
		Program    artifact.ID                      `json:"program"`
	}{
		Recipe: spec.Recipe, Mode: mode, Model: model.String(), Checkpoint: checkpoint.String(),
		Scratch: scratch.ID().String(), Dataset: spec.Dataset, Split: spec.Split,
		Signature: signature, Processors: processors, Projectors: projectors,
		Codecs: codecs, Policies: spec.Policies, Program: spec.Program.ID(),
	}
	id, err := artifact.JSONID(artifact.KindRecipe, body)
	if err != nil {
		return TrainingRunPlan{}, err
	}
	return TrainingRunPlan{
		id: id, recipe: spec.Recipe, mode: mode, model: model, checkpoint: checkpoint,
		scratch: scratch, dataset: spec.Dataset, split: spec.Split, signature: signature,
		processors: processors, projectors: projectors, codecs: codecs,
		policies: spec.Policies, program: spec.Program,
	}, nil
}

func (p TrainingRunPlan) ID() artifact.ID                             { return p.id }
func (p TrainingRunPlan) InitialMode() InitialStateMode               { return p.mode }
func (p TrainingRunPlan) Model() artifact.ID                          { return p.model }
func (p TrainingRunPlan) Dataset() artifact.ID                        { return p.dataset }
func (p TrainingRunPlan) Split() artifact.ID                          { return p.split }
func (p TrainingRunPlan) Program() TrainingProgram                    { return p.program }
func (p TrainingRunPlan) Signature() recipecontract.ModalitySignature { return p.signature.Clone() }
func (p TrainingRunPlan) Processors() []artifact.ID                   { return slices.Clone(p.processors) }
func (p TrainingRunPlan) Projectors() []artifact.ID                   { return slices.Clone(p.projectors) }
func (p TrainingRunPlan) Codecs() []artifact.ID                       { return slices.Clone(p.codecs) }

func compileRNGStreams(source []RNGStreamSpec) ([]RNGStreamSpec, error) {
	streams := slices.Clone(source)
	sort.Slice(streams, func(i, j int) bool { return streams[i].Name < streams[j].Name })
	required := []string{"augmentation", "data", "init", "split"}
	if len(streams) != len(required) {
		return nil, errors.New("training program: scratch construction requires split/init/data/augmentation RNG streams")
	}
	for index, stream := range streams {
		if stream.Name != required[index] || stream.Algorithm.Kind() != artifact.KindProfile {
			return nil, errors.New("training program: invalid RNG stream")
		}
	}
	return streams, nil
}

func compileOperators(source []OperatorSpec) ([]OperatorSpec, error) {
	operators := slices.Clone(source)
	if len(operators) < 3 {
		return nil, errors.New("training program: forward/backward/optimize operators required")
	}
	seen := make(map[string]struct{}, len(operators))
	positions := map[OperatorPhase]int{}
	for index, operator := range operators {
		if strings.TrimSpace(operator.ID) == "" || strings.ContainsAny(operator.ID, "\x00\r\n") || !validPhase(operator.Phase) {
			return nil, errors.New("training program: invalid operator")
		}
		if _, duplicate := seen[operator.ID]; duplicate {
			return nil, errors.New("training program: duplicate operator")
		}
		seen[operator.ID] = struct{}{}
		if _, exists := positions[operator.Phase]; !exists {
			positions[operator.Phase] = index
		}
	}
	forward, hasForward := positions[PhaseForward]
	backward, hasBackward := positions[PhaseBackward]
	optimize, hasOptimize := positions[PhaseOptimize]
	if !hasForward || !hasBackward || !hasOptimize || forward >= backward || backward >= optimize {
		return nil, errors.New("training program: semantic operator order differs")
	}
	return operators, nil
}

func validPhase(phase OperatorPhase) bool {
	switch phase {
	case PhaseBatch, PhaseForward, PhaseLoss, PhaseBackward, PhaseOptimize, PhaseEvaluate:
		return true
	default:
		return false
	}
}

func compileParameters(source []ParameterSpec, plan optimizer.Plan) ([]ParameterSpec, error) {
	if plan.Identity() == "" || len(source) != plan.GroupCount() {
		return nil, errors.New("training program: parameter manifest differs from Muon plan")
	}
	parameters := slices.Clone(source)
	for index, parameter := range parameters {
		group, _ := plan.Group(index)
		if parameter.Name != group.Name || parameter.Rows != group.Rows || parameter.Cols != group.Cols || parameter.Trainable == group.Frozen {
			return nil, fmt.Errorf("training program: parameter %d differs from Muon group", index)
		}
	}
	return parameters, nil
}

func compileInitialState(spec InitialStateSpec) (InitialStateMode, artifact.ID, artifact.ID, ScratchConstruction, error) {
	count := 0
	if spec.Model.Valid() {
		count++
	}
	if spec.Checkpoint.Valid() {
		count++
	}
	if spec.Scratch != nil {
		count++
	}
	if count != 1 {
		return "", artifact.ID{}, artifact.ID{}, ScratchConstruction{}, errors.New("training program: exactly one initial state is required")
	}
	if spec.Model.Valid() {
		if spec.Model.Kind() != artifact.KindModel {
			return "", artifact.ID{}, artifact.ID{}, ScratchConstruction{}, errors.New("training program: initial model kind differs")
		}
		return InitialPretrained, spec.Model, artifact.ID{}, ScratchConstruction{}, nil
	}
	if spec.Checkpoint.Valid() {
		if spec.Checkpoint.Kind() != artifact.KindCheckpoint {
			return "", artifact.ID{}, artifact.ID{}, ScratchConstruction{}, errors.New("training program: resume checkpoint kind differs")
		}
		return InitialResume, artifact.ID{}, spec.Checkpoint, ScratchConstruction{}, nil
	}
	if spec.Scratch.ID().Kind() != artifact.KindRecipe {
		return "", artifact.ID{}, artifact.ID{}, ScratchConstruction{}, errors.New("training program: scratch construction is not compiled")
	}
	return InitialScratch, spec.Scratch.InitializedModel(), artifact.ID{}, *spec.Scratch, nil
}

func compileIDs(label string, source []artifact.ID, kind artifact.Kind) ([]artifact.ID, error) {
	result := slices.Clone(source)
	sort.Slice(result, func(i, j int) bool { return result[i].String() < result[j].String() })
	for index, id := range result {
		if id.Kind() != kind || index > 0 && result[index-1] == id {
			return nil, fmt.Errorf("training program: invalid %s identities", label)
		}
	}
	return result, nil
}

func validatePolicies(spec PolicySpec) error {
	for _, id := range []artifact.ID{spec.Objective, spec.Precision, spec.Placement, spec.Memory, spec.Checkpoint, spec.Evaluation, spec.Promotion} {
		if id.Kind() != artifact.KindProfile {
			return errors.New("training program: invalid policy identity")
		}
	}
	return nil
}
