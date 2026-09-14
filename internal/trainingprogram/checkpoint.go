package trainingprogram

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/hostoptimizer"
	"overgo/internal/strictjson"
)

const (
	CheckpointFilename = "checkpoint.json"
	CheckpointWeights  = "model.safetensors"
)

type RNGState struct {
	Name      string      `json:"name"`
	Algorithm artifact.ID `json:"algorithm"`
	Seed      uint64      `json:"seed"`
	Counter   uint64      `json:"counter"`
}

type LineageParent struct {
	Artifact artifact.ID       `json:"artifact"`
	Relation artifact.Relation `json:"relation"`
}

type DatasetState struct {
	Identity artifact.ID `json:"identity"`
	Position uint64      `json:"position"`
}

// RouterControllerState is the exact next-step selection state for one routed layer and dataset stratum.
type RouterControllerState struct {
	Policy   artifact.ID `json:"policy"`
	Stratum  artifact.ID `json:"stratum"`
	Layer    int         `json:"layer"`
	NextBias []float32   `json:"next_bias"`
}

// RouterControllerAuthority describes the policy and geometry allowed to resume stored router state.
type RouterControllerAuthority struct {
	Policy  artifact.ID
	Stratum artifact.ID
	Layer   int
	Experts int
}

type CheckpointSpec struct {
	RunPlan           artifact.ID
	Program           artifact.ID
	Model             artifact.ID
	Dataset           artifact.ID
	Split             artifact.ID
	Stream            DatasetState
	Optimizer         hostoptimizer.State
	ParameterCount    int
	RNG               []RNGState
	Processors        []artifact.ID
	Projectors        []artifact.ID
	Codecs            []artifact.ID
	Lineage           []LineageParent
	RouterControllers []RouterControllerState
	Accumulation      int
}

type Checkpoint struct {
	id                artifact.ID
	RunPlan           artifact.ID             `json:"run_plan"`
	Program           artifact.ID             `json:"program"`
	Model             artifact.ID             `json:"model"`
	Weights           artifact.ID             `json:"weights"`
	Dataset           artifact.ID             `json:"dataset"`
	Split             artifact.ID             `json:"split"`
	Stream            DatasetState            `json:"stream"`
	Optimizer         hostoptimizer.State     `json:"optimizer"`
	ParameterCount    int                     `json:"parameter_count"`
	RNG               []RNGState              `json:"rng"`
	Processors        []artifact.ID           `json:"processors"`
	Projectors        []artifact.ID           `json:"projectors,omitempty"`
	Codecs            []artifact.ID           `json:"codecs,omitempty"`
	Lineage           []LineageParent         `json:"lineage"`
	RouterControllers []RouterControllerState `json:"router_controllers,omitempty"`
	Accumulation      int                     `json:"accumulation"`
}

type checkpointDocument struct {
	Version uint16 `json:"version"`
	Checkpoint
}

func (c Checkpoint) ID() artifact.ID { return c.id }

// Content returns the canonical checkpoint document with its existing identity.
// Weight bytes remain in the separately identified Safetensors artifact.
func (c Checkpoint) Content() (artifact.Content, error) {
	if err := c.ValidateIdentity(); err != nil {
		return artifact.Content{}, err
	}
	return artifact.JSONContent(artifact.JSONContract(artifact.KindCheckpoint, "overgo/training-checkpoint/v1"),
		checkpointDocument{Version: artifact.InitialDocumentVersion, Checkpoint: c})
}

// Batch binds a complete on-disk checkpoint to its canonical document, weights,
// lineage and locations. It does not copy weights or activate a model recipe.
func (c Checkpoint) Batch(key, directory string) (artifact.Batch, error) {
	loaded, err := LoadCheckpoint(directory)
	if err != nil || loaded.ID() != c.ID() {
		return artifact.Batch{}, errors.Join(errors.New("training checkpoint: publication differs"), err)
	}
	content, err := c.Content()
	if err != nil {
		return artifact.Batch{}, err
	}
	weightsPath := filepath.Join(directory, CheckpointWeights)
	info, err := os.Stat(weightsPath)
	if err != nil {
		return artifact.Batch{}, err
	}
	location, err := artifact.CanonicalLocalLocation(c.ID(), artifact.LocationDirectory, directory)
	if err != nil {
		return artifact.Batch{}, err
	}
	weightsLocation, err := artifact.CanonicalLocalLocation(c.Weights, artifact.LocationFile, weightsPath)
	if err != nil {
		return artifact.Batch{}, err
	}
	batch, err := artifact.NewDocumentBatch(key, []artifact.Content{content}, c.ArtifactLineage(), nil)
	if err != nil {
		return artifact.Batch{}, err
	}
	batch.Artifacts = []artifact.Descriptor{{ID: c.Weights, Size: uint64(info.Size()), MediaType: "application/vnd.safetensors"}}
	batch.Lineage = append(batch.Lineage, artifact.Lineage{Child: c.ID(), Parent: c.Weights, Relation: artifact.RelationContains})
	batch.Locations = []artifact.LocationEvent{{Location: location, Action: artifact.LocationAdd}, {Location: weightsLocation, Action: artifact.LocationAdd}}
	return batch, batch.Validate()
}

func (c Checkpoint) ValidateIdentity() error {
	copy := c
	copy.id = artifact.ID{}
	copy.Optimizer.Momentum = slices.Clone(c.Optimizer.Momentum)
	copy.RNG = slices.Clone(c.RNG)
	copy.Processors = slices.Clone(c.Processors)
	copy.Projectors = slices.Clone(c.Projectors)
	copy.Codecs = slices.Clone(c.Codecs)
	copy.Lineage = slices.Clone(c.Lineage)
	copy.RouterControllers = cloneRouterControllerStates(c.RouterControllers)
	if err := canonicalizeCheckpoint(&copy); err != nil {
		return err
	}
	want, err := artifact.JSONID(
		artifact.KindCheckpoint,
		checkpointDocument{Version: artifact.InitialDocumentVersion, Checkpoint: copy},
	)
	if err != nil || want != c.id {
		return errors.Join(err, errors.New("training checkpoint: identity differs"))
	}
	return nil
}

func (c Checkpoint) ArtifactLineage() []artifact.Lineage {
	result := make([]artifact.Lineage, len(c.Lineage))
	seen := make(map[artifact.Lineage]struct{}, len(c.Lineage)+len(c.RouterControllers)*2)
	for index, parent := range c.Lineage {
		result[index] = artifact.Lineage{Child: c.id, Parent: parent.Artifact, Relation: parent.Relation}
		seen[result[index]] = struct{}{}
	}
	for _, controller := range c.RouterControllers {
		for _, parent := range []artifact.ID{controller.Policy, controller.Stratum} {
			edge := artifact.Lineage{Child: c.id, Parent: parent, Relation: artifact.RelationDependsOn}
			if _, found := seen[edge]; found {
				continue
			}
			seen[edge] = struct{}{}
			result = append(result, edge)
		}
	}
	return result
}

func NewCheckpoint(spec CheckpointSpec, weights artifact.ID) (Checkpoint, error) {
	checkpoint := Checkpoint{
		RunPlan: spec.RunPlan, Program: spec.Program, Model: spec.Model, Weights: weights,
		Dataset: spec.Dataset, Split: spec.Split, Stream: spec.Stream,
		Optimizer: spec.Optimizer, ParameterCount: spec.ParameterCount,
		RNG: slices.Clone(spec.RNG), Processors: slices.Clone(spec.Processors),
		Projectors: slices.Clone(spec.Projectors), Codecs: slices.Clone(spec.Codecs),
		Lineage: slices.Clone(spec.Lineage), RouterControllers: cloneRouterControllerStates(spec.RouterControllers), Accumulation: spec.Accumulation,
	}
	if err := canonicalizeCheckpoint(&checkpoint); err != nil {
		return Checkpoint{}, err
	}
	id, err := artifact.JSONID(
		artifact.KindCheckpoint,
		checkpointDocument{Version: artifact.InitialDocumentVersion, Checkpoint: checkpoint})

	if err != nil {
		return Checkpoint{}, err
	}
	checkpoint.id = id
	return checkpoint, nil
}

func (c Checkpoint) Marshal() ([]byte, error) {
	copy := c
	copy.id = artifact.ID{}
	if err := canonicalizeCheckpoint(&copy); err != nil {
		return nil, err
	}
	return json.MarshalIndent(checkpointDocument{Version: artifact.InitialDocumentVersion, Checkpoint: copy}, "", "  ")
}

func ParseCheckpoint(data []byte) (Checkpoint, error) {
	var document checkpointDocument
	if err := strictjson.DecodeBytes(data, &document); err != nil {
		return Checkpoint{}, fmt.Errorf("training checkpoint: decode: %w", err)
	}
	if document.Version != artifact.InitialDocumentVersion {
		return Checkpoint{}, errors.New("training checkpoint: version differs")
	}
	return NewCheckpoint(CheckpointSpec{
		RunPlan: document.RunPlan, Program: document.Program, Model: document.Model,
		Dataset: document.Dataset, Split: document.Split, Stream: document.Stream,
		Optimizer: document.Optimizer, ParameterCount: document.ParameterCount,
		RNG: document.RNG, Processors: document.Processors, Projectors: document.Projectors,
		Codecs: document.Codecs, Lineage: document.Lineage, RouterControllers: document.RouterControllers, Accumulation: document.Accumulation,
	}, document.Weights)
}

func PublishCheckpoint(target string, spec CheckpointSpec, writeWeights func(string) error) (Checkpoint, error) {
	if strings.TrimSpace(target) == "" || writeWeights == nil {
		return Checkpoint{}, errors.New("training checkpoint: target and weight writer are required")
	}
	target, err := filepath.Abs(target)
	if err != nil {
		return Checkpoint{}, err
	}
	if _, err := os.Stat(target); err == nil {
		return Checkpoint{}, errors.New("training checkpoint: target already exists")
	} else if !errors.Is(err, os.ErrNotExist) {
		return Checkpoint{}, err
	}
	parent := filepath.Dir(target)
	if err := os.MkdirAll(parent, 0o755); err != nil {
		return Checkpoint{}, err
	}
	stage, err := os.MkdirTemp(parent, ".overgo-checkpoint-*")
	if err != nil {
		return Checkpoint{}, err
	}
	committed := false
	defer func() {
		if !committed {
			_ = os.RemoveAll(stage)
		}
	}()
	if err := writeWeights(stage); err != nil {
		return Checkpoint{}, err
	}
	weightsPath := filepath.Join(stage, CheckpointWeights)
	weightsFile, err := os.Open(weightsPath)
	if err != nil {
		return Checkpoint{}, fmt.Errorf("training checkpoint: open weights: %w", err)
	}
	weightsID, _, identifyErr := artifact.Identify(artifact.KindTensorSet, weightsFile)
	closeErr := weightsFile.Close()
	if err := errors.Join(identifyErr, closeErr); err != nil {
		return Checkpoint{}, err
	}
	checkpoint, err := NewCheckpoint(spec, weightsID)
	if err != nil {
		return Checkpoint{}, err
	}
	data, err := checkpoint.Marshal()
	if err != nil {
		return Checkpoint{}, err
	}
	if err := os.WriteFile(filepath.Join(stage, CheckpointFilename), append(data, '\n'), 0o644); err != nil {
		return Checkpoint{}, err
	}
	if err := os.Rename(stage, target); err != nil {
		return Checkpoint{}, fmt.Errorf("training checkpoint: publish: %w", err)
	}
	committed = true
	return checkpoint, nil
}

func LoadCheckpoint(directory string) (Checkpoint, error) {
	data, err := os.ReadFile(filepath.Join(directory, CheckpointFilename))
	if err != nil {
		return Checkpoint{}, err
	}
	checkpoint, err := ParseCheckpoint(data)
	if err != nil {
		return Checkpoint{}, err
	}
	file, err := os.Open(filepath.Join(directory, CheckpointWeights))
	if err != nil {
		return Checkpoint{}, err
	}
	weights, _, identifyErr := artifact.Identify(artifact.KindTensorSet, file)
	closeErr := file.Close()
	if err := errors.Join(identifyErr, closeErr); err != nil {
		return Checkpoint{}, err
	}
	if weights != checkpoint.Weights {
		return Checkpoint{}, errors.New("training checkpoint: weights identity differs")
	}
	return checkpoint, nil
}

type ResumeAuthority struct {
	Model             artifact.ID
	Stream            DatasetState
	OptimizerPlan     string
	RouterControllers []RouterControllerAuthority
}

func ValidateResume(plan TrainingRunPlan, checkpoint Checkpoint, authority ResumeAuthority) error {
	if err := checkpoint.ValidateIdentity(); err != nil {
		return err
	}
	checkpointControllers := make([]RouterControllerAuthority, len(checkpoint.RouterControllers))
	for index, state := range checkpoint.RouterControllers {
		checkpointControllers[index] = routerControllerAuthority(state)
	}
	if plan.InitialMode() != InitialResume || plan.checkpoint != checkpoint.ID() ||
		plan.Program().ID() != checkpoint.Program || plan.Program().OptimizerIdentity() != checkpoint.Optimizer.PlanIdentity ||
		authority.Model != checkpoint.Model || authority.Stream != checkpoint.Stream ||
		authority.OptimizerPlan != checkpoint.Optimizer.PlanIdentity ||
		plan.Dataset() != checkpoint.Dataset || plan.Split() != checkpoint.Split ||
		!slices.Equal(plan.Processors(), checkpoint.Processors) ||
		!slices.Equal(plan.Projectors(), checkpoint.Projectors) || !slices.Equal(plan.Codecs(), checkpoint.Codecs) ||
		!slices.Equal(canonicalRouterControllerAuthorities(authority.RouterControllers), checkpointControllers) {
		return errors.New("training checkpoint: resume authority differs")
	}
	return nil
}

func canonicalizeCheckpoint(checkpoint *Checkpoint) error {
	if checkpoint == nil || checkpoint.RunPlan.Kind() != artifact.KindRecipe ||
		checkpoint.Program.Kind() != artifact.KindRecipe || checkpoint.Model.Kind() != artifact.KindModel ||
		checkpoint.Weights.Kind() != artifact.KindTensorSet || checkpoint.Dataset.Kind() != artifact.KindDataset ||
		checkpoint.Split.Kind() != artifact.KindDatasetShard || !checkpoint.Stream.Identity.Valid() ||
		checkpoint.ParameterCount <= 0 || checkpoint.Accumulation != 0 {
		return errors.New("training checkpoint: invalid authority or accumulation boundary")
	}
	if err := hostoptimizer.ValidateState(checkpoint.Optimizer, checkpoint.Optimizer.PlanIdentity, checkpoint.ParameterCount); err != nil {
		return fmt.Errorf("training checkpoint: %w", err)
	}
	if err := canonicalRNG(&checkpoint.RNG); err != nil {
		return err
	}
	for label, ids := range map[string]*[]artifact.ID{
		"processor": &checkpoint.Processors, "projector": &checkpoint.Projectors, "codec": &checkpoint.Codecs,
	} {
		kind := artifact.KindProfile
		if label == "projector" {
			kind = artifact.KindProjector
		}
		if err := canonicalCheckpointIDs(label, ids, kind); err != nil {
			return err
		}
	}
	if len(checkpoint.Processors) == 0 || len(checkpoint.Lineage) == 0 {
		return errors.New("training checkpoint: processor or lineage authority absent")
	}
	slices.SortFunc(checkpoint.Lineage, func(left, right LineageParent) int {
		if order := cmp.Compare(left.Artifact.String(), right.Artifact.String()); order != 0 {
			return order
		}
		return cmp.Compare(left.Relation, right.Relation)
	})
	for index, parent := range checkpoint.Lineage {
		if !parent.Artifact.Valid() || parent.Relation == artifact.RelationInvalid ||
			index > 0 && parent == checkpoint.Lineage[index-1] {
			return errors.New("training checkpoint: invalid lineage")
		}
	}
	if err := canonicalRouterControllerStates(&checkpoint.RouterControllers); err != nil {
		return err
	}
	return nil
}

func cloneRouterControllerStates(states []RouterControllerState) []RouterControllerState {
	copy := slices.Clone(states)
	for index := range copy {
		copy[index].NextBias = slices.Clone(copy[index].NextBias)
	}
	return copy
}

func canonicalRouterControllerStates(states *[]RouterControllerState) error {
	slices.SortFunc(*states, func(left, right RouterControllerState) int {
		return compareRouterControllerAuthority(routerControllerAuthority(left), routerControllerAuthority(right))
	})
	for index, state := range *states {
		if state.Policy.Kind() != artifact.KindRecipe || state.Stratum.Kind() != artifact.KindDatasetShard ||
			state.Layer < 0 || len(state.NextBias) == 0 ||
			index > 0 && (*states)[index-1].Policy == state.Policy &&
				(*states)[index-1].Stratum == state.Stratum && (*states)[index-1].Layer == state.Layer {
			return errors.New("training checkpoint: invalid router controller state")
		}
		for _, bias := range state.NextBias {
			if math.IsNaN(float64(bias)) || math.IsInf(float64(bias), 0) {
				return errors.New("training checkpoint: non-finite router controller state")
			}
		}
	}
	return nil
}

func routerControllerAuthority(state RouterControllerState) RouterControllerAuthority {
	return RouterControllerAuthority{Policy: state.Policy, Stratum: state.Stratum, Layer: state.Layer, Experts: len(state.NextBias)}
}

func canonicalRouterControllerAuthorities(authorities []RouterControllerAuthority) []RouterControllerAuthority {
	result := slices.Clone(authorities)
	slices.SortFunc(result, compareRouterControllerAuthority)
	return result
}

func compareRouterControllerAuthority(left, right RouterControllerAuthority) int {
	if left.Policy != right.Policy {
		return artifact.CompareID(left.Policy, right.Policy)
	}
	if left.Stratum != right.Stratum {
		return artifact.CompareID(left.Stratum, right.Stratum)
	}
	return cmp.Compare(left.Layer, right.Layer)
}

func canonicalRNG(states *[]RNGState) error {
	slices.SortFunc(*states, func(left, right RNGState) int { return cmp.Compare(left.Name, right.Name) })
	required := map[string]bool{"augmentation": false, "data": false}
	for index, state := range *states {
		if strings.TrimSpace(state.Name) == "" || strings.ContainsAny(state.Name, "\x00\r\n") ||
			state.Algorithm.Kind() != artifact.KindProfile || index > 0 && state.Name == (*states)[index-1].Name {
			return errors.New("training checkpoint: invalid RNG state")
		}
		if _, ok := required[state.Name]; ok {
			required[state.Name] = true
		}
	}
	if !required["augmentation"] || !required["data"] {
		return errors.New("training checkpoint: data and augmentation RNG state required")
	}
	return nil
}

func canonicalCheckpointIDs(label string, ids *[]artifact.ID, kind artifact.Kind) error {
	slices.SortFunc(*ids, func(left, right artifact.ID) int { return cmp.Compare(left.String(), right.String()) })
	for index, id := range *ids {
		if id.Kind() != kind || index > 0 && id == (*ids)[index-1] {
			return fmt.Errorf("training checkpoint: invalid %s identities", label)
		}
	}
	return nil
}
