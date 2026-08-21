package trainingprogram

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/optimizer"
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

type CheckpointSpec struct {
	RunPlan        artifact.ID
	Program        artifact.ID
	Model          artifact.ID
	Dataset        artifact.ID
	Split          artifact.ID
	Stream         DatasetState
	Optimizer      optimizer.State
	ParameterCount int
	RNG            []RNGState
	Processors     []artifact.ID
	Projectors     []artifact.ID
	Codecs         []artifact.ID
	Lineage        []LineageParent
	Accumulation   int
}

type Checkpoint struct {
	id             artifact.ID
	RunPlan        artifact.ID     `json:"run_plan"`
	Program        artifact.ID     `json:"program"`
	Model          artifact.ID     `json:"model"`
	Weights        artifact.ID     `json:"weights"`
	Dataset        artifact.ID     `json:"dataset"`
	Split          artifact.ID     `json:"split"`
	Stream         DatasetState    `json:"stream"`
	Optimizer      optimizer.State `json:"optimizer"`
	ParameterCount int             `json:"parameter_count"`
	RNG            []RNGState      `json:"rng"`
	Processors     []artifact.ID   `json:"processors"`
	Projectors     []artifact.ID   `json:"projectors,omitempty"`
	Codecs         []artifact.ID   `json:"codecs,omitempty"`
	Lineage        []LineageParent `json:"lineage"`
	Accumulation   int             `json:"accumulation"`
}

type checkpointDocument struct {
	Version uint16 `json:"version"`
	Checkpoint
}

func (c Checkpoint) ID() artifact.ID { return c.id }

func (c Checkpoint) ArtifactLineage() []artifact.Lineage {
	result := make([]artifact.Lineage, len(c.Lineage))
	for index, parent := range c.Lineage {
		result[index] = artifact.Lineage{Child: c.id, Parent: parent.Artifact, Relation: parent.Relation}
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
		Lineage: slices.Clone(spec.Lineage), Accumulation: spec.Accumulation,
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
		Codecs: document.Codecs, Lineage: document.Lineage, Accumulation: document.Accumulation,
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

func ValidateResume(plan TrainingRunPlan, checkpoint Checkpoint, stream artifact.ID) error {
	if plan.InitialMode() != InitialResume || !checkpoint.ID().Valid() || plan.checkpoint != checkpoint.ID() ||
		plan.Program().ID() != checkpoint.Program || plan.Dataset() != checkpoint.Dataset || plan.Split() != checkpoint.Split ||
		stream != checkpoint.Stream.Identity || !slices.Equal(plan.Processors(), checkpoint.Processors) ||
		!slices.Equal(plan.Projectors(), checkpoint.Projectors) || !slices.Equal(plan.Codecs(), checkpoint.Codecs) {
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
	if err := optimizer.ValidateState(checkpoint.Optimizer, checkpoint.Optimizer.PlanIdentity, checkpoint.ParameterCount); err != nil {
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
	sort.Slice(checkpoint.Lineage, func(i, j int) bool {
		if checkpoint.Lineage[i].Artifact != checkpoint.Lineage[j].Artifact {
			return checkpoint.Lineage[i].Artifact.String() < checkpoint.Lineage[j].Artifact.String()
		}
		return checkpoint.Lineage[i].Relation < checkpoint.Lineage[j].Relation
	})
	for index, parent := range checkpoint.Lineage {
		if !parent.Artifact.Valid() || parent.Relation == artifact.RelationInvalid ||
			index > 0 && parent == checkpoint.Lineage[index-1] {
			return errors.New("training checkpoint: invalid lineage")
		}
	}
	return nil
}

func canonicalRNG(states *[]RNGState) error {
	sort.Slice(*states, func(i, j int) bool { return (*states)[i].Name < (*states)[j].Name })
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
	sort.Slice(*ids, func(i, j int) bool { return (*ids)[i].String() < (*ids)[j].String() })
	for index, id := range *ids {
		if id.Kind() != kind || index > 0 && id == (*ids)[index-1] {
			return fmt.Errorf("training checkpoint: invalid %s identities", label)
		}
	}
	return nil
}
