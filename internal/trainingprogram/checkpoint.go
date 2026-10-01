package trainingprogram

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"unsafe"

	"overgo/internal/artifact"
	"overgo/internal/binaryschema"
	"overgo/internal/hostoptimizer"
	"overgo/internal/strictjson"
)

const (
	CheckpointFilename = "checkpoint.json"
	CheckpointWeights  = "model.safetensors"
	// CheckpointModel is the trained model a run over a GGUF base publishes
	// in that base's layout, the model the server opens. A checkpoint that
	// holds it binds it as its weights in place of CheckpointWeights.
	CheckpointModel = "model.gguf"
	// resumeFilename holds the resume sidecar's document: the optimizer
	// progress, dataset stream position and RNG only a continued run reads.
	resumeFilename = "resume.json"
	// momentumFilename holds the resume sidecar's momentum as raw
	// little-endian values in the dtype the trainer steps.
	momentumFilename = "momentum.bin"
	// momentumF32 and momentumF64 name the dtypes a momentum file holds.
	momentumF32 = "f32"
	momentumF64 = "f64"
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
	id      artifact.ID
	RunPlan artifact.ID `json:"run_plan"`
	Program artifact.ID `json:"program"`
	Model   artifact.ID `json:"model"`
	Weights artifact.ID `json:"weights"`
	// Resume names the resume sidecar, which names Weights in turn.
	Resume            artifact.ID             `json:"resume"`
	Dataset           artifact.ID             `json:"dataset"`
	Split             artifact.ID             `json:"split"`
	ParameterCount    int                     `json:"parameter_count"`
	Processors        []artifact.ID           `json:"processors"`
	Projectors        []artifact.ID           `json:"projectors,omitempty"`
	Codecs            []artifact.ID           `json:"codecs,omitempty"`
	Lineage           []LineageParent         `json:"lineage"`
	RouterControllers []RouterControllerState `json:"router_controllers,omitempty"`
	Accumulation      int                     `json:"accumulation"`
	// Stream, Optimizer and RNG only continue the run: the resume sidecar
	// holds them, and a checkpoint read without it leaves them empty.
	Stream    DatasetState        `json:"-"`
	Optimizer hostoptimizer.State `json:"-"`
	RNG       []RNGState          `json:"-"`
	// momentum identifies the sidecar's momentum file; it is valid only
	// while the sidecar's state is held.
	momentum artifact.ID
}

type checkpointDocument struct {
	Version uint16 `json:"version"`
	Checkpoint
}

// resumeDocument is the resume sidecar's document, bound by digest to the
// weights it continues and to its momentum file.
type resumeDocument struct {
	Version      uint16               `json:"version"`
	Weights      artifact.ID          `json:"weights"`
	PlanIdentity string               `json:"plan_identity"`
	Config       hostoptimizer.Config `json:"config"`
	Step         int                  `json:"step"`
	Momentum     artifact.ID          `json:"momentum"`
	DType        string               `json:"dtype"`
	Stream       DatasetState         `json:"stream"`
	RNG          []RNGState           `json:"rng"`
}

func (c Checkpoint) ID() artifact.ID { return c.id }

var (
	// checkpointContract is the canonical checkpoint document's contract.
	checkpointContract = artifact.JSONContract(artifact.KindCheckpoint, "overgo/training-checkpoint/v2")
	// resumeContract is the resume sidecar document's contract.
	resumeContract = artifact.JSONContract(artifact.KindCheckpoint, "overgo/training-resume/v1")
)

// Content returns the canonical checkpoint document with its existing identity.
// Weight bytes and the resume sidecar remain separately identified.
func (c Checkpoint) Content() (artifact.Content, error) {
	if err := c.ValidateIdentity(); err != nil {
		return artifact.Content{}, err
	}
	return artifact.JSONContent(checkpointContract, c.document())
}

func (c Checkpoint) document() checkpointDocument {
	return checkpointDocument{Version: artifact.SecondDocumentVersion, Checkpoint: c}
}

func (c Checkpoint) resume() resumeDocument {
	_, dtype := momentumBytes(c.Optimizer)
	return resumeDocument{
		Version: artifact.InitialDocumentVersion, Weights: c.Weights,
		PlanIdentity: c.Optimizer.PlanIdentity, Config: c.Optimizer.Config, Step: c.Optimizer.Step,
		Momentum: c.momentum, DType: dtype, Stream: c.Stream, RNG: c.RNG,
	}
}

// Batch binds a complete on-disk checkpoint to its canonical document, its
// resume sidecar, weights, lineage and locations, each file identified by
// its digest. It does not copy weights or activate a model recipe.
func (c Checkpoint) Batch(key, directory string) (artifact.Batch, error) {
	if err := c.ValidateIdentity(); err != nil {
		return artifact.Batch{}, err
	}
	var contents []artifact.Content
	for _, document := range []struct {
		name     string
		contract artifact.DocumentContract
		id       artifact.ID
	}{{CheckpointFilename, checkpointContract, c.id}, {resumeFilename, resumeContract, c.Resume}} {
		data, err := os.ReadFile(filepath.Join(directory, document.name))
		if err != nil {
			return artifact.Batch{}, err
		}
		content, err := document.contract.ContentBytes(data)
		if err != nil || content.Descriptor.ID != document.id {
			return artifact.Batch{}, errors.Join(errors.New("training checkpoint: publication differs"), err)
		}
		contents = append(contents, content)
	}
	location, err := artifact.CanonicalLocalLocation(c.ID(), artifact.LocationDirectory, directory)
	if err != nil {
		return artifact.Batch{}, err
	}
	batch, err := artifact.NewDocumentBatch(key, contents, c.ArtifactLineage(), nil)
	if err != nil {
		return artifact.Batch{}, err
	}
	batch.Locations = []artifact.LocationEvent{{Location: location, Action: artifact.LocationAdd}}
	weights, weightsType := weightsFile(directory)
	for _, file := range []struct {
		path, mediaType string
		id              artifact.ID
	}{{weights, weightsType, c.Weights}, {filepath.Join(directory, momentumFilename), "application/octet-stream", c.momentum}} {
		info, err := os.Stat(file.path)
		if err != nil {
			return artifact.Batch{}, err
		}
		fileLocation, err := artifact.CanonicalLocalLocation(file.id, artifact.LocationFile, file.path)
		if err != nil {
			return artifact.Batch{}, err
		}
		batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: file.id, Size: uint64(info.Size()), MediaType: file.mediaType})
		batch.Locations = append(batch.Locations, artifact.LocationEvent{Location: fileLocation, Action: artifact.LocationAdd})
	}
	batch.Lineage = append(batch.Lineage,
		artifact.Lineage{Child: c.ID(), Parent: c.Weights, Relation: artifact.RelationContains},
		artifact.Lineage{Child: c.ID(), Parent: c.Resume, Relation: artifact.RelationDependsOn},
		artifact.Lineage{Child: c.Resume, Parent: c.Weights, Relation: artifact.RelationDependsOn},
		artifact.Lineage{Child: c.Resume, Parent: c.momentum, Relation: artifact.RelationContains},
	)
	return batch, batch.Validate()
}

// ValidateIdentity recomputes the checkpoint's identity, and its resume
// sidecar's while the sidecar's state is held.
func (c Checkpoint) ValidateIdentity() error {
	copy := c
	copy.RNG = slices.Clone(c.RNG)
	copy.Processors = slices.Clone(c.Processors)
	copy.Projectors = slices.Clone(c.Projectors)
	copy.Codecs = slices.Clone(c.Codecs)
	copy.Lineage = slices.Clone(c.Lineage)
	copy.RouterControllers = cloneRouterControllerStates(c.RouterControllers)
	if err := identify(&copy); err != nil || copy.Resume != c.Resume || copy.id != c.id {
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

// identify canonicalizes checkpoint and sets its document identity, and its
// resume sidecar's while the sidecar's state is held.
func identify(checkpoint *Checkpoint) error {
	if err := canonicalizeCheckpoint(checkpoint); err != nil {
		return err
	}
	if checkpoint.momentum.Valid() {
		if !checkpoint.Stream.Identity.Valid() {
			return errors.New("training checkpoint: invalid stream authority")
		}
		if err := canonicalRNG(&checkpoint.RNG); err != nil {
			return err
		}
		resume, err := artifact.JSONID(artifact.KindCheckpoint, checkpoint.resume())
		if err != nil {
			return err
		}
		checkpoint.Resume = resume
	}
	if checkpoint.Resume.Kind() != artifact.KindCheckpoint {
		return errors.New("training checkpoint: resume sidecar identity absent")
	}
	id, err := artifact.JSONID(artifact.KindCheckpoint, checkpoint.document())
	checkpoint.id = id
	return err
}

// ParseCheckpoint reads a checkpoint document without its resume sidecar.
func ParseCheckpoint(data []byte) (Checkpoint, error) {
	var document checkpointDocument
	if err := json.Unmarshal(data, &document); err != nil || document.Version != artifact.SecondDocumentVersion {
		return Checkpoint{}, errors.Join(fmt.Errorf("training checkpoint: not a version %d document, which keeps its momentum, stream and RNG in the resume sidecar; an earlier version inlined them and is republished from its run", artifact.SecondDocumentVersion), err)
	}
	if err := strictjson.DecodeBytes(data, &document); err != nil {
		return Checkpoint{}, fmt.Errorf("training checkpoint: decode: %w", err)
	}
	checkpoint := document.Checkpoint
	if err := identify(&checkpoint); err != nil {
		return Checkpoint{}, err
	}
	return checkpoint, nil
}

// PublishCheckpoint publishes, at a fresh target, the weights writeWeights
// stages with the checkpoint document and resume sidecar that bind them.
func PublishCheckpoint(target string, spec CheckpointSpec, writeWeights func(string) error) (Checkpoint, error) {
	if strings.TrimSpace(target) == "" || writeWeights == nil {
		return Checkpoint{}, errors.New("training checkpoint: target and weight writer are required")
	}
	if err := hostoptimizer.ValidateState(spec.Optimizer, spec.Optimizer.PlanIdentity, spec.ParameterCount); err != nil {
		return Checkpoint{}, fmt.Errorf("training checkpoint: %w", err)
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
	weights, _ := weightsFile(stage)
	file, err := os.Open(weights)
	if err != nil {
		return Checkpoint{}, fmt.Errorf("training checkpoint: open weights: %w", err)
	}
	weightsID, _, identifyErr := artifact.Identify(artifact.KindTensorSet, file)
	closeErr := file.Close()
	if err := errors.Join(identifyErr, closeErr); err != nil {
		return Checkpoint{}, err
	}
	// The momentum is hashed and streamed to disk from the trainer's own
	// buffer.
	values, _ := momentumBytes(spec.Optimizer)
	momentum, err := artifact.IdentifyBytes(artifact.KindFile, values)
	if err != nil {
		return Checkpoint{}, err
	}
	checkpoint := Checkpoint{
		RunPlan: spec.RunPlan, Program: spec.Program, Model: spec.Model, Weights: weightsID,
		Dataset: spec.Dataset, Split: spec.Split, Stream: spec.Stream,
		Optimizer: spec.Optimizer, ParameterCount: spec.ParameterCount,
		RNG: slices.Clone(spec.RNG), Processors: slices.Clone(spec.Processors),
		Projectors: slices.Clone(spec.Projectors), Codecs: slices.Clone(spec.Codecs),
		Lineage: slices.Clone(spec.Lineage), RouterControllers: cloneRouterControllerStates(spec.RouterControllers),
		Accumulation: spec.Accumulation, momentum: momentum,
	}
	if err := identify(&checkpoint); err != nil {
		return Checkpoint{}, err
	}
	// Each document is written as the canonical bytes its identity hashes.
	files := map[string][]byte{momentumFilename: values}
	for name, document := range map[string]any{CheckpointFilename: checkpoint.document(), resumeFilename: checkpoint.resume()} {
		if files[name], err = json.Marshal(document); err != nil {
			return Checkpoint{}, err
		}
	}
	for name, data := range files {
		if err := os.WriteFile(filepath.Join(stage, name), data, 0o644); err != nil {
			return Checkpoint{}, err
		}
	}
	if err := os.Rename(stage, target); err != nil {
		return Checkpoint{}, fmt.Errorf("training checkpoint: publish: %w", err)
	}
	committed = true
	return checkpoint, nil
}

// LoadCheckpoint reads a published checkpoint with its resume sidecar,
// holding each file to the digest the document chain names.
func LoadCheckpoint(directory string) (Checkpoint, error) {
	data, err := os.ReadFile(filepath.Join(directory, CheckpointFilename))
	if err != nil {
		return Checkpoint{}, err
	}
	checkpoint, err := ParseCheckpoint(data)
	if err != nil {
		return Checkpoint{}, err
	}
	weightsPath, _ := weightsFile(directory)
	file, err := os.Open(weightsPath)
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
	data, err = os.ReadFile(filepath.Join(directory, resumeFilename))
	if err != nil {
		return Checkpoint{}, fmt.Errorf("training checkpoint: resume sidecar: %w", err)
	}
	var resume resumeDocument
	if err := strictjson.DecodeBytes(data, &resume); err != nil {
		return Checkpoint{}, fmt.Errorf("training checkpoint: decode resume sidecar: %w", err)
	}
	if resume.Version != artifact.InitialDocumentVersion || resume.Weights != checkpoint.Weights {
		return Checkpoint{}, errors.New("training checkpoint: resume sidecar is bound to other weights")
	}
	// The state is the one its publication validated: the digests chain from
	// the checkpoint identity to every momentum byte.
	state, err := readMomentum(filepath.Join(directory, momentumFilename), resume, checkpoint.ParameterCount)
	if err != nil {
		return Checkpoint{}, err
	}
	checkpoint.Stream, checkpoint.RNG, checkpoint.Optimizer, checkpoint.momentum = resume.Stream, resume.RNG, state, resume.Momentum
	return checkpoint, checkpoint.ValidateIdentity()
}

// momentumBytes views the momentum state holds as its little-endian bytes
// in place, and names its dtype: the momentum file is written and read
// through the view, so streaming it never holds a second copy. Every
// target overgo builds for is little-endian.
func momentumBytes(state hostoptimizer.State) ([]byte, string) {
	if len(state.MomentumF32) != 0 {
		return unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(state.MomentumF32))), len(state.MomentumF32)*binaryschema.Uint32Bytes), momentumF32
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(state.Momentum))), len(state.Momentum)*binaryschema.Uint64Bytes), momentumF64
}

// weightsFile is the weights file a checkpoint directory holds and its
// media type: the trained model when the run published one, else the
// trained tensors.
func weightsFile(directory string) (string, string) {
	model := filepath.Join(directory, CheckpointModel)
	if _, err := os.Stat(model); err == nil {
		return model, "application/vnd.gguf"
	}
	return filepath.Join(directory, CheckpointWeights), "application/vnd.safetensors"
}

// readMomentum reads the momentum file at path, in place, into the
// optimizer state resume names, holding the file to its digest and to
// parameters values.
func readMomentum(path string, resume resumeDocument, parameters int) (hostoptimizer.State, error) {
	state := hostoptimizer.State{PlanIdentity: resume.PlanIdentity, Config: resume.Config, Step: resume.Step}
	switch resume.DType {
	case momentumF32:
		state.MomentumF32 = make([]float32, parameters)
	case momentumF64:
		state.Momentum = make([]float64, parameters)
	default:
		return state, fmt.Errorf("training checkpoint: momentum dtype %q is none a trainer steps", resume.DType)
	}
	data, _ := momentumBytes(state)
	file, err := os.Open(path)
	if err != nil {
		return state, fmt.Errorf("training checkpoint: resume sidecar: %w", err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || info.Size() != int64(len(data)) {
		return state, errors.Join(fmt.Errorf("training checkpoint: the momentum file does not hold %d %s values", parameters, resume.DType), err)
	}
	if _, err := io.ReadFull(file, data); err != nil {
		return state, err
	}
	if id, err := artifact.IdentifyBytes(artifact.KindFile, data); err != nil || id != resume.Momentum {
		return state, errors.Join(errors.New("training checkpoint: momentum file differs from its resume sidecar"), err)
	}
	return state, nil
}

type ResumeAuthority struct {
	Model             artifact.ID
	Stream            DatasetState
	OptimizerPlan     string
	RouterControllers []RouterControllerAuthority
}

func ValidateResume(plan TrainingRunPlan, checkpoint Checkpoint, authority ResumeAuthority) error {
	if !checkpoint.momentum.Valid() {
		return errors.New("training checkpoint: resume needs the checkpoint's resume sidecar")
	}
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
		checkpoint.Split.Kind() != artifact.KindDatasetShard ||
		checkpoint.ParameterCount <= 0 || checkpoint.Accumulation != 0 {
		return errors.New("training checkpoint: invalid authority or accumulation boundary")
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
