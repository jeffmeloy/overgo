package runrecord

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
	"overgo/internal/textcheck"
)

const (
	GateVersion   uint16 = 1
	GateMediaType        = "application/vnd.overgo.gate-result+json"
	GateSchema           = "overgo/gate-result/v1"
)

var gateContract = artifact.DocumentContract{
	Kind: artifact.KindEvidence, MediaType: GateMediaType, Schema: GateSchema,
}

var gateCodec = artifact.DocumentCodec[GateResult]{
	Name: "run record gate result", Contract: gateContract,
	Decode: func(data []byte, value *GateResult) error { return strictjson.DecodeBytes(data, value) },
	Encode: gateContent, Canonicalize: canonicalizeGateResult,
	Clone: func(value GateResult) GateResult {
		value.Steps = slices.Clone(value.Steps)
		return value
	},
	Identity:    func(value GateResult) artifact.ID { return value.ID },
	SetIdentity: func(value *GateResult, id artifact.ID) { value.ID = id },
}

type StepOutcome string

const (
	StepSucceeded StepOutcome = "succeeded"
	StepFailed    StepOutcome = "failed"
	StepCancelled StepOutcome = "cancelled"
	StepSkipped   StepOutcome = "skipped"
)

type GateStep struct {
	Name       string      `json:"name"`
	Phase      Phase       `json:"phase"`
	Outcome    StepOutcome `json:"outcome"`
	DurationNS uint64      `json:"duration_ns,omitempty"`
	Evidence   string      `json:"evidence,omitempty"`
}

// GateResult: immutable named-step gate verdict.
type GateResult struct {
	Version     uint16      `json:"version"`
	Recipe      artifact.ID `json:"recipe"`
	Environment artifact.ID `json:"environment"`
	CodeCommit  string      `json:"code_commit"`
	Outcome     Outcome     `json:"outcome"`
	Failure     string      `json:"failure,omitempty"`
	Steps       []GateStep  `json:"steps"`
	ID          artifact.ID `json:"-"`
}

type GateRecord struct {
	Result GateResult
	Run    Run
}

func NewGateRecord(
	recipeID, environment artifact.ID,
	codeCommit string,
	outcome Outcome,
	failure string,
	measuredNS uint64,
	steps []GateStep,
) (GateRecord, error) {
	result := GateResult{
		Version: GateVersion, Recipe: recipeID, Environment: environment,
		CodeCommit: codeCommit, Outcome: outcome, Failure: failure, Steps: slices.Clone(steps),
	}
	result, err := gateCodec.New(result)
	if err != nil {
		return GateRecord{}, err
	}
	phases, err := aggregateGatePhases(result.Steps)
	if err != nil {
		return GateRecord{}, err
	}
	run, err := NewBoundRun(
		recipeID, outcome, nil, []artifact.ID{result.ID}, failure, codeCommit,
		environment, measuredNS, phases,
	)
	if err != nil {
		return GateRecord{}, err
	}
	return GateRecord{Result: result, Run: run}, nil
}

func ParseGateResult(content []byte) (GateResult, error) {
	return gateCodec.Parse(content)
}

func (g GateResult) ValidateIdentity() error {
	return gateCodec.ValidateIdentity(g)
}

func (g GateResult) Content() (artifact.Content, error) {
	return gateCodec.Content(g)
}

func (g GateResult) Lineage() []artifact.Lineage {
	return []artifact.Lineage{
		{Child: g.ID, Parent: g.Recipe, Relation: artifact.RelationDependsOn},
		{Child: g.ID, Parent: g.Environment, Relation: artifact.RelationDependsOn},
	}
}

func (g GateRecord) Batch(key string) (artifact.Batch, error) {
	resultContent, err := g.Result.Content()
	if err != nil {
		return artifact.Batch{}, err
	}
	runContent, err := g.Run.Content()
	if err != nil {
		return artifact.Batch{}, err
	}
	lineage := append(g.Result.Lineage(), g.Run.Lineage()...)
	return artifact.NewDocumentBatch(key, []artifact.Content{resultContent, runContent}, lineage, nil)
}

func canonicalizeGateResult(result *GateResult) error {
	if result == nil || result.Version != GateVersion || result.Recipe.Kind() != artifact.KindRecipe ||
		result.Environment.Kind() != artifact.KindEvidence || !validCodeCommit(result.CodeCommit) ||
		len(result.Steps) == 0 || len(result.Steps) > MaxAdvisoryWindow {
		return errors.New("run record: invalid gate result")
	}
	switch result.Outcome {
	case OutcomeSucceeded:
		if result.Failure != "" {
			return errors.New("run record: successful gate has failure code")
		}
	case OutcomeFailed:
		if !validLabel(result.Failure) {
			return errors.New("run record: failed gate lacks failure code")
		}
	case OutcomeCancelled:
		if result.Failure != "" {
			return errors.New("run record: cancelled gate has failure code")
		}
	default:
		return errors.New("run record: invalid gate outcome")
	}
	seen := make(map[string]struct{}, len(result.Steps))
	terminalMatch := false
	for _, step := range result.Steps {
		_, duplicate := seen[step.Name]
		switch {
		case !validLabel(step.Name):
			return fmt.Errorf("run record: invalid gate step name %q", step.Name)
		case !validPhase(step.Phase):
			return fmt.Errorf("run record: gate step %q has invalid phase", step.Name)
		case step.Evidence != "" && !textcheck.Bounded(step.Evidence, 2048, "\x00\r\n"):
			return fmt.Errorf("run record: gate step %q has invalid evidence", step.Name)
		case step.DurationNS > math.MaxInt64 || step.Outcome != StepSkipped && step.DurationNS == 0:
			return fmt.Errorf("run record: gate step %q has invalid duration", step.Name)
		case duplicate:
			return fmt.Errorf("run record: duplicate gate step %q", step.Name)
		}
		seen[step.Name] = struct{}{}
		switch step.Outcome {
		case StepSucceeded, StepSkipped:
		case StepFailed:
			terminalMatch = terminalMatch || result.Outcome == OutcomeFailed
		case StepCancelled:
			terminalMatch = terminalMatch || result.Outcome == OutcomeCancelled
		default:
			return errors.New("run record: invalid gate step outcome")
		}
		if result.Outcome == OutcomeSucceeded && step.Outcome != StepSucceeded && step.Outcome != StepSkipped {
			return errors.New("run record: successful gate has terminal step")
		}
	}
	if result.Outcome != OutcomeSucceeded && !terminalMatch {
		return errors.New("run record: gate outcome lacks matching terminal step")
	}
	return nil
}

func aggregateGatePhases(steps []GateStep) ([]PhaseMetric, error) {
	totals := make(map[Phase]uint64)
	for _, step := range steps {
		if step.DurationNS == 0 {
			continue
		}
		if math.MaxInt64-totals[step.Phase] < step.DurationNS {
			return nil, errors.New("run record: gate phase duration overflow")
		}
		totals[step.Phase] += step.DurationNS
	}
	phases := make([]PhaseMetric, 0, len(totals))
	for phase, duration := range totals {
		phases = append(phases, PhaseMetric{Phase: phase, DurationNS: duration})
	}
	sort.Slice(phases, func(i, j int) bool { return phases[i].Phase < phases[j].Phase })
	return phases, nil
}

func gateContent(result GateResult) ([]byte, error) {
	result.ID = artifact.ID{}
	content, err := json.Marshal(result)
	if err != nil {
		return nil, fmt.Errorf("run record: encode gate result: %w", err)
	}
	return content, nil
}
