package runrecord

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
	"overgo/internal/textcheck"
)

const (
	Version             uint16 = 1
	RunVersion          uint16 = 2
	RunMediaType               = "application/vnd.overgo.run+json"
	LegacyRunSchema            = "overgo/run/v1"
	RunSchema                  = "overgo/run/v2"
	EvaluationMediaType        = "application/vnd.overgo.evaluation+json"
	EvaluationSchema           = "overgo/evaluation/v1"
	maxLabelBytes              = 128
)

var (
	runContract = artifact.DocumentContract{
		Kind: artifact.KindRun, MediaType: RunMediaType, Schema: RunSchema,
	}
	legacyRunContract = artifact.DocumentContract{
		Kind: artifact.KindRun, MediaType: RunMediaType, Schema: LegacyRunSchema,
	}
	evaluationContract = artifact.DocumentContract{
		Kind: artifact.KindEvaluation, MediaType: EvaluationMediaType, Schema: EvaluationSchema,
	}
)

var evaluationCodec = artifact.DocumentCodec[Evaluation]{
	Name: "run record evaluation", Contract: evaluationContract,
	Decode: func(data []byte, value *Evaluation) error {
		var body evaluationBody
		if err := strictjson.DecodeBytes(data, &body); err != nil {
			return err
		}
		*value = Evaluation{
			Version: body.Version, Recipe: body.Recipe, Run: body.Run,
			Dataset: body.Dataset, Metrics: body.Metrics,
		}
		return nil
	},
	Encode: evaluationContent, Canonicalize: canonicalizeEvaluation,
	Clone: func(value Evaluation) Evaluation {
		value.Metrics = slices.Clone(value.Metrics)
		return value
	},
	Identity:    func(value Evaluation) artifact.ID { return value.ID },
	SetIdentity: func(value *Evaluation, id artifact.ID) { value.ID = id },
}

var runCodec = artifact.DocumentCodec[Run]{
	Name: "run record", ContractFor: func(value Run) artifact.DocumentContract {
		return runContractForVersion(value.Version)
	},
	Decode: func(data []byte, value *Run) error {
		var body runBody
		if err := strictjson.DecodeBytes(data, &body); err != nil {
			return err
		}
		*value = Run{
			Version: body.Version, Recipe: body.Recipe, Outcome: body.Outcome,
			Inputs: body.Inputs, Outputs: body.Outputs, Failure: body.Failure,
			CodeCommit: body.CodeCommit, MeasuredNS: body.MeasuredNS, Phases: body.Phases,
		}
		if body.Environment != nil {
			value.Environment = *body.Environment
		}
		return nil
	},
	Encode: runContent, Canonicalize: canonicalizeRun, Clone: cloneRun,
	Identity:    func(value Run) artifact.ID { return value.ID },
	SetIdentity: func(value *Run, id artifact.ID) { value.ID = id },
}

type Outcome string

const (
	OutcomeSucceeded Outcome = "succeeded"
	OutcomeFailed    Outcome = "failed"
	OutcomeCancelled Outcome = "cancelled"
)

type Direction string

const (
	DirectionNeutral  Direction = "neutral"
	DirectionMinimize Direction = "minimize"
	DirectionMaximize Direction = "maximize"
)

type Run struct {
	Version     uint16
	ID          artifact.ID
	Recipe      artifact.ID
	Outcome     Outcome
	Inputs      []artifact.ID
	Outputs     []artifact.ID
	Failure     string
	CodeCommit  string
	Environment artifact.ID
	MeasuredNS  uint64
	Phases      []PhaseMetric
}

type Phase string

const (
	PhaseLoad            Phase = "load"
	PhasePromptRender    Phase = "prompt_render"
	PhaseTokenize        Phase = "tokenize"
	PhaseMediaDecode     Phase = "media_decode"
	PhaseVision          Phase = "vision"
	PhasePrefill         Phase = "prefill"
	PhaseDecode          Phase = "decode"
	PhaseSample          Phase = "sample"
	PhaseForwardBackward Phase = "forward_backward"
	PhaseOptimizer       Phase = "optimizer"
	PhaseRefresh         Phase = "refresh"
	PhaseHostToDevice    Phase = "host_to_device"
	PhaseDeviceToHost    Phase = "device_to_host"
	PhasePostprocess     Phase = "postprocess"
	PhaseValidate        Phase = "validate"
	PhaseBuild           Phase = "build"
	PhaseTest            Phase = "test"
	PhaseVet             Phase = "vet"
	PhasePackage         Phase = "package"
)

type PhaseMetric struct {
	Phase      Phase  `json:"phase"`
	DurationNS uint64 `json:"duration_ns"`
}

type Metric struct {
	Name      string    `json:"name"`
	Value     float64   `json:"value"`
	Unit      string    `json:"unit,omitempty"`
	Direction Direction `json:"direction"`
}

type Evaluation struct {
	Version uint16
	ID      artifact.ID
	Recipe  artifact.ID
	Run     artifact.ID
	Dataset artifact.ID
	Metrics []Metric
}

type runBody struct {
	Version     uint16        `json:"version"`
	Recipe      artifact.ID   `json:"recipe"`
	Outcome     Outcome       `json:"outcome"`
	Inputs      []artifact.ID `json:"inputs,omitempty"`
	Outputs     []artifact.ID `json:"outputs,omitempty"`
	Failure     string        `json:"failure,omitempty"`
	CodeCommit  string        `json:"code_commit,omitempty"`
	Environment *artifact.ID  `json:"environment,omitempty"`
	MeasuredNS  uint64        `json:"measured_ns,omitempty"`
	Phases      []PhaseMetric `json:"phases,omitempty"`
}

type evaluationBody struct {
	Version uint16      `json:"version"`
	Recipe  artifact.ID `json:"recipe"`
	Run     artifact.ID `json:"run"`
	Dataset artifact.ID `json:"dataset"`
	Metrics []Metric    `json:"metrics"`
}

func NewRun(
	recipeID artifact.ID,
	outcome Outcome,
	inputs, outputs []artifact.ID,
	failure string,
) (Run, error) {
	return runCodec.New(Run{
		Version: Version, Recipe: recipeID, Outcome: outcome,
		Inputs: slices.Clone(inputs), Outputs: slices.Clone(outputs), Failure: failure,
	})
}

func NewBoundRun(
	recipeID artifact.ID,
	outcome Outcome,
	inputs, outputs []artifact.ID,
	failure, codeCommit string,
	environment artifact.ID,
	measuredNS uint64,
	phases []PhaseMetric,
) (Run, error) {
	return runCodec.New(Run{
		Version: RunVersion, Recipe: recipeID, Outcome: outcome,
		Inputs: slices.Clone(inputs), Outputs: slices.Clone(outputs), Failure: failure,
		CodeCommit: codeCommit, Environment: environment, MeasuredNS: measuredNS,
		Phases: slices.Clone(phases),
	})
}

func ParseRun(content []byte) (Run, error) {
	return runCodec.Parse(content)
}

func (r Run) ValidateIdentity() error {
	return runCodec.ValidateIdentity(r)
}

func (r Run) Content() (artifact.Content, error) {
	return runCodec.Content(r)
}

func (r Run) Lineage() []artifact.Lineage {
	edges := []artifact.Lineage{{
		Child: r.ID, Parent: r.Recipe, Relation: artifact.RelationDependsOn,
	}}
	if r.Version == RunVersion {
		edges = append(edges, artifact.Lineage{
			Child: r.ID, Parent: r.Environment, Relation: artifact.RelationDependsOn,
		})
	}
	for _, input := range r.Inputs {
		edges = append(edges, artifact.Lineage{
			Child: r.ID, Parent: input, Relation: artifact.RelationDependsOn,
		})
	}
	for _, output := range r.Outputs {
		edges = append(edges, artifact.Lineage{
			Child: output, Parent: r.ID, Relation: artifact.RelationProducedBy,
		})
	}
	return edges
}

func (r Run) Batch(key string) (artifact.Batch, error) {
	content, err := r.Content()
	if err != nil {
		return artifact.Batch{}, err
	}
	return artifact.NewDocumentBatch(key, []artifact.Content{content}, r.Lineage(), nil)
}

func NewEvaluation(
	recipeID, runID, datasetID artifact.ID,
	metrics []Metric,
) (Evaluation, error) {
	evaluation := Evaluation{
		Version: Version, Recipe: recipeID, Run: runID, Dataset: datasetID,
		Metrics: slices.Clone(metrics),
	}
	return evaluationCodec.New(evaluation)
}

func ParseEvaluation(content []byte) (Evaluation, error) {
	return evaluationCodec.Parse(content)
}

func (e Evaluation) ValidateIdentity() error {
	return evaluationCodec.ValidateIdentity(e)
}

func (e Evaluation) Content() (artifact.Content, error) {
	return evaluationCodec.Content(e)
}

func (e Evaluation) Lineage() []artifact.Lineage {
	return []artifact.Lineage{
		{Child: e.ID, Parent: e.Recipe, Relation: artifact.RelationDependsOn},
		{Child: e.ID, Parent: e.Run, Relation: artifact.RelationDependsOn},
		{Child: e.ID, Parent: e.Dataset, Relation: artifact.RelationDependsOn},
	}
}

func (e Evaluation) Batch(key string) (artifact.Batch, error) {
	content, err := e.Content()
	if err != nil {
		return artifact.Batch{}, err
	}
	return artifact.NewDocumentBatch(key, []artifact.Content{content}, e.Lineage(), nil)
}

func canonicalizeRun(run *Run) error {
	if run == nil || run.Version != Version && run.Version != RunVersion || run.Recipe.Kind() != artifact.KindRecipe {
		return errors.New("run record: invalid run envelope")
	}
	if run.Version == Version {
		if run.CodeCommit != "" || run.Environment.Valid() || run.MeasuredNS != 0 || len(run.Phases) != 0 {
			return errors.New("run record: legacy run carries bound facts")
		}
	} else if !validCodeCommit(run.CodeCommit) || run.Environment.Kind() != artifact.KindEvidence ||
		run.MeasuredNS == 0 || run.MeasuredNS > math.MaxInt64 {
		return errors.New("run record: invalid bound run facts")
	}
	if run.Outcome != OutcomeSucceeded && run.Outcome != OutcomeFailed && run.Outcome != OutcomeCancelled {
		return errors.New("run record: invalid outcome")
	}
	if run.Outcome == OutcomeSucceeded && (len(run.Outputs) == 0 || run.Failure != "") {
		return errors.New("run record: invalid successful outcome")
	}
	if run.Outcome == OutcomeFailed && !textcheck.LowerIdentifier(run.Failure, maxLabelBytes) {
		return errors.New("run record: failed outcome requires failure code")
	}
	if run.Outcome == OutcomeCancelled && run.Failure != "" {
		return errors.New("run record: cancelled outcome has failure code")
	}
	if err := canonicalIDs(run.Inputs); err != nil {
		return fmt.Errorf("run record: inputs: %w", err)
	}
	if err := canonicalIDs(run.Outputs); err != nil {
		return fmt.Errorf("run record: outputs: %w", err)
	}
	if err := canonicalizePhases(&run.Phases); err != nil {
		return err
	}
	return nil
}

func canonicalizePhases(phases *[]PhaseMetric) error {
	sort.Slice(*phases, func(i, j int) bool { return (*phases)[i].Phase < (*phases)[j].Phase })
	var total uint64
	for index, metric := range *phases {
		if !validPhase(metric.Phase) || metric.DurationNS == 0 || metric.DurationNS > math.MaxInt64 ||
			index > 0 && (*phases)[index-1].Phase == metric.Phase || total > math.MaxInt64-metric.DurationNS {
			return errors.New("run record: invalid phase metric")
		}
		total += metric.DurationNS
	}
	return nil
}

func validPhase(phase Phase) bool {
	switch phase {
	case PhaseLoad, PhasePromptRender, PhaseTokenize, PhaseMediaDecode, PhaseVision,
		PhasePrefill, PhaseDecode, PhaseSample, PhaseForwardBackward, PhaseOptimizer,
		PhaseRefresh, PhaseHostToDevice, PhaseDeviceToHost, PhasePostprocess:
		return true
	case PhaseValidate, PhaseBuild, PhaseTest, PhaseVet, PhasePackage:
		return true
	default:
		return false
	}
}

func validCodeCommit(value string) bool {
	if len(value) != 40 && len(value) != 64 {
		return false
	}
	for _, character := range value {
		if character < '0' || character > '9' && character < 'a' || character > 'f' {
			return false
		}
	}
	return true
}

func canonicalizeEvaluation(evaluation *Evaluation) error {
	if evaluation == nil || evaluation.Version != Version ||
		evaluation.Recipe.Kind() != artifact.KindRecipe || evaluation.Run.Kind() != artifact.KindRun ||
		evaluation.Dataset.Kind() != artifact.KindDataset || len(evaluation.Metrics) == 0 {
		return errors.New("run record: invalid evaluation envelope")
	}
	for _, metric := range evaluation.Metrics {
		if !textcheck.LowerIdentifier(metric.Name, maxLabelBytes) || len(metric.Unit) > maxLabelBytes ||
			strings.TrimSpace(metric.Unit) != metric.Unit || strings.ContainsAny(metric.Unit, "\r\n") ||
			math.IsNaN(metric.Value) || math.IsInf(metric.Value, 0) ||
			metric.Direction != DirectionNeutral && metric.Direction != DirectionMinimize &&
				metric.Direction != DirectionMaximize {
			return errors.New("run record: invalid metric")
		}
	}
	sort.Slice(evaluation.Metrics, func(i, j int) bool {
		return evaluation.Metrics[i].Name < evaluation.Metrics[j].Name
	})
	for index := 1; index < len(evaluation.Metrics); index++ {
		if evaluation.Metrics[index-1].Name == evaluation.Metrics[index].Name {
			return errors.New("run record: duplicate metric")
		}
	}
	return nil
}

func canonicalIDs(ids []artifact.ID) error {
	for _, id := range ids {
		if !id.Valid() {
			return errors.New("invalid artifact identity")
		}
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i].String() < ids[j].String() })
	for index := 1; index < len(ids); index++ {
		if ids[index-1] == ids[index] {
			return errors.New("duplicate artifact identity")
		}
	}
	return nil
}

func runContent(run Run) ([]byte, error) {
	var environment *artifact.ID
	if run.Version == RunVersion {
		environment = artifact.IDPointer(run.Environment)
	}
	content, err := json.Marshal(runBody{
		Version: run.Version, Recipe: run.Recipe, Outcome: run.Outcome,
		Inputs: run.Inputs, Outputs: run.Outputs, Failure: run.Failure,
		CodeCommit: run.CodeCommit, Environment: environment,
		MeasuredNS: run.MeasuredNS, Phases: run.Phases,
	})
	if err != nil {
		return nil, fmt.Errorf("run record: encode run: %w", err)
	}
	return content, nil
}

func evaluationContent(evaluation Evaluation) ([]byte, error) {
	content, err := json.Marshal(evaluationBody{
		Version: evaluation.Version, Recipe: evaluation.Recipe, Run: evaluation.Run,
		Dataset: evaluation.Dataset, Metrics: evaluation.Metrics,
	})
	if err != nil {
		return nil, fmt.Errorf("run record: encode evaluation: %w", err)
	}
	return content, nil
}

func cloneRun(run Run) Run {
	run.Inputs = slices.Clone(run.Inputs)
	run.Outputs = slices.Clone(run.Outputs)
	run.Phases = slices.Clone(run.Phases)
	return run
}

func runContractForVersion(version uint16) artifact.DocumentContract {
	if version == Version {
		return legacyRunContract
	}
	return runContract
}
