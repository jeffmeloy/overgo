package runrecord

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

const (
	GateVersion   uint16 = 1
	GateMediaType        = "application/vnd.overgo.gate-result+json"
	GateSchema           = "overgo/gate-result/v1"
)

var gateContract = artifact.DocumentContract{
	Kind: artifact.KindEvidence, MediaType: GateMediaType, Schema: GateSchema,
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
	if err := canonicalizeGateResult(&result); err != nil {
		return GateRecord{}, err
	}
	content, err := gateContent(result)
	if err != nil {
		return GateRecord{}, err
	}
	result.ID, err = gateContract.Identify(content)
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
	var result GateResult
	if err := strictjson.DecodeBytes(content, &result); err != nil {
		return GateResult{}, fmt.Errorf("run record: decode gate result: %w", err)
	}
	result.ID = artifact.ID{}
	if err := canonicalizeGateResult(&result); err != nil {
		return GateResult{}, err
	}
	canonical, err := gateContent(result)
	if err != nil {
		return GateResult{}, err
	}
	if !bytes.Equal(canonical, content) {
		return GateResult{}, errors.New("run record: non-canonical gate result")
	}
	result.ID, err = gateContract.Identify(canonical)
	return result, err
}

func (g GateResult) ValidateIdentity() error {
	if g.ID.Kind() != artifact.KindEvidence {
		return errors.New("run record: invalid gate result identity")
	}
	canonical := g
	canonical.ID = artifact.ID{}
	canonical.Steps = slices.Clone(g.Steps)
	if err := canonicalizeGateResult(&canonical); err != nil {
		return err
	}
	canonical.ID = g.ID
	if g.Version != canonical.Version || g.Recipe != canonical.Recipe ||
		g.Environment != canonical.Environment || g.CodeCommit != canonical.CodeCommit ||
		g.Outcome != canonical.Outcome || g.Failure != canonical.Failure || !slices.Equal(g.Steps, canonical.Steps) {
		return errors.New("run record: gate result is not canonical")
	}
	canonical.ID = artifact.ID{}
	content, err := gateContent(canonical)
	if err != nil {
		return err
	}
	if err := gateContract.ValidateIdentity(g.ID, content); err != nil {
		return errors.New("run record: gate result identity mismatch")
	}
	return nil
}

func (g GateResult) ContentBytes() ([]byte, error) {
	if err := g.ValidateIdentity(); err != nil {
		return nil, err
	}
	g.ID = artifact.ID{}
	return gateContent(g)
}

func (g GateResult) Content() (artifact.Content, error) {
	id := g.ID
	content, err := g.ContentBytes()
	if err != nil {
		return artifact.Content{}, err
	}
	return gateContract.Content(id, content)
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
		len(result.Steps) == 0 || len(result.Steps) > maxAdvisoryWindow {
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
		if !validLabel(step.Name) || !validPhase(step.Phase) ||
			step.DurationNS > math.MaxInt64 || step.Outcome != StepSkipped && step.DurationNS == 0 ||
			duplicate {
			return errors.New("run record: invalid gate step")
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
