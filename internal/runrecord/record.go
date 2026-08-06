package runrecord

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"
	"strings"

	"llamacpp2go/internal/artifact"
	"llamacpp2go/internal/strictjson"
)

const (
	Version             uint16 = 1
	RunMediaType               = "application/vnd.llamacpp2go.run+json"
	RunSchema                  = "llamacpp2go/run/v1"
	EvaluationMediaType        = "application/vnd.llamacpp2go.evaluation+json"
	EvaluationSchema           = "llamacpp2go/evaluation/v1"
	maxLabelBytes              = 128
)

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
	Version uint16
	ID      artifact.ID
	Recipe  artifact.ID
	Outcome Outcome
	Inputs  []artifact.ID
	Outputs []artifact.ID
	Failure string
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
	Version uint16        `json:"version"`
	Recipe  artifact.ID   `json:"recipe"`
	Outcome Outcome       `json:"outcome"`
	Inputs  []artifact.ID `json:"inputs,omitempty"`
	Outputs []artifact.ID `json:"outputs,omitempty"`
	Failure string        `json:"failure,omitempty"`
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
	run := Run{
		Version: Version, Recipe: recipeID, Outcome: outcome,
		Inputs: slices.Clone(inputs), Outputs: slices.Clone(outputs), Failure: failure,
	}
	if err := canonicalizeRun(&run); err != nil {
		return Run{}, err
	}
	content, err := runContent(run)
	if err != nil {
		return Run{}, err
	}
	run.ID, err = artifact.IdentifyBytes(artifact.KindRun, content)
	return run, err
}

func ParseRun(content []byte) (Run, error) {
	var body runBody
	if err := strictjson.DecodeBytes(content, &body); err != nil {
		return Run{}, fmt.Errorf("run record: decode run: %w", err)
	}
	run, err := NewRun(body.Recipe, body.Outcome, body.Inputs, body.Outputs, body.Failure)
	if err != nil {
		return Run{}, err
	}
	canonical, err := run.ContentBytes()
	if err != nil {
		return Run{}, err
	}
	if !bytes.Equal(canonical, content) {
		return Run{}, errors.New("run record: non-canonical run content")
	}
	return run, nil
}

func (r Run) ValidateIdentity() error {
	if r.ID.Kind() != artifact.KindRun {
		return errors.New("run record: invalid run identity")
	}
	canonical := r
	if err := canonicalizeRun(&canonical); err != nil {
		return err
	}
	if !sameRun(r, canonical) {
		return errors.New("run record: run is not canonical")
	}
	content, err := runContent(canonical)
	if err != nil {
		return err
	}
	want, err := artifact.IdentifyBytes(artifact.KindRun, content)
	if err != nil {
		return err
	}
	if r.ID != want {
		return errors.New("run record: run identity mismatch")
	}
	return nil
}

func (r Run) ContentBytes() ([]byte, error) {
	if err := r.ValidateIdentity(); err != nil {
		return nil, err
	}
	return runContent(r)
}

func (r Run) Content() (artifact.Content, error) {
	content, err := r.ContentBytes()
	if err != nil {
		return artifact.Content{}, err
	}
	return artifact.Content{Descriptor: artifact.Descriptor{
		ID: r.ID, Size: uint64(len(content)), MediaType: RunMediaType, Schema: RunSchema,
	}, Data: content}, nil
}

func (r Run) Lineage() []artifact.Lineage {
	edges := []artifact.Lineage{{
		Child: r.ID, Parent: r.Recipe, Relation: artifact.RelationDependsOn,
	}}
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
	return artifact.Batch{Key: key, Contents: []artifact.Content{content}, Lineage: r.Lineage()}, nil
}

func NewEvaluation(
	recipeID, runID, datasetID artifact.ID,
	metrics []Metric,
) (Evaluation, error) {
	evaluation := Evaluation{
		Version: Version, Recipe: recipeID, Run: runID, Dataset: datasetID,
		Metrics: slices.Clone(metrics),
	}
	if err := canonicalizeEvaluation(&evaluation); err != nil {
		return Evaluation{}, err
	}
	content, err := evaluationContent(evaluation)
	if err != nil {
		return Evaluation{}, err
	}
	evaluation.ID, err = artifact.IdentifyBytes(artifact.KindEvaluation, content)
	return evaluation, err
}

func ParseEvaluation(content []byte) (Evaluation, error) {
	var body evaluationBody
	if err := strictjson.DecodeBytes(content, &body); err != nil {
		return Evaluation{}, fmt.Errorf("run record: decode evaluation: %w", err)
	}
	evaluation, err := NewEvaluation(body.Recipe, body.Run, body.Dataset, body.Metrics)
	if err != nil {
		return Evaluation{}, err
	}
	canonical, err := evaluation.ContentBytes()
	if err != nil {
		return Evaluation{}, err
	}
	if !bytes.Equal(canonical, content) {
		return Evaluation{}, errors.New("run record: non-canonical evaluation content")
	}
	return evaluation, nil
}

func (e Evaluation) ValidateIdentity() error {
	if e.ID.Kind() != artifact.KindEvaluation {
		return errors.New("run record: invalid evaluation identity")
	}
	canonical := e
	if err := canonicalizeEvaluation(&canonical); err != nil {
		return err
	}
	if !sameEvaluation(e, canonical) {
		return errors.New("run record: evaluation is not canonical")
	}
	content, err := evaluationContent(canonical)
	if err != nil {
		return err
	}
	want, err := artifact.IdentifyBytes(artifact.KindEvaluation, content)
	if err != nil {
		return err
	}
	if e.ID != want {
		return errors.New("run record: evaluation identity mismatch")
	}
	return nil
}

func (e Evaluation) ContentBytes() ([]byte, error) {
	if err := e.ValidateIdentity(); err != nil {
		return nil, err
	}
	return evaluationContent(e)
}

func (e Evaluation) Content() (artifact.Content, error) {
	content, err := e.ContentBytes()
	if err != nil {
		return artifact.Content{}, err
	}
	return artifact.Content{Descriptor: artifact.Descriptor{
		ID: e.ID, Size: uint64(len(content)), MediaType: EvaluationMediaType, Schema: EvaluationSchema,
	}, Data: content}, nil
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
	return artifact.Batch{Key: key, Contents: []artifact.Content{content}, Lineage: e.Lineage()}, nil
}

func canonicalizeRun(run *Run) error {
	if run == nil || run.Version != Version || run.Recipe.Kind() != artifact.KindRecipe {
		return errors.New("run record: invalid run envelope")
	}
	if run.Outcome != OutcomeSucceeded && run.Outcome != OutcomeFailed && run.Outcome != OutcomeCancelled {
		return errors.New("run record: invalid outcome")
	}
	if run.Outcome == OutcomeSucceeded && (len(run.Outputs) == 0 || run.Failure != "") {
		return errors.New("run record: invalid successful outcome")
	}
	if run.Outcome == OutcomeFailed && !validLabel(run.Failure) {
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
	return nil
}

func canonicalizeEvaluation(evaluation *Evaluation) error {
	if evaluation == nil || evaluation.Version != Version ||
		evaluation.Recipe.Kind() != artifact.KindRecipe || evaluation.Run.Kind() != artifact.KindRun ||
		evaluation.Dataset.Kind() != artifact.KindDataset || len(evaluation.Metrics) == 0 {
		return errors.New("run record: invalid evaluation envelope")
	}
	for _, metric := range evaluation.Metrics {
		if !validLabel(metric.Name) || len(metric.Unit) > maxLabelBytes ||
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

func validLabel(value string) bool {
	if value == "" || len(value) > maxLabelBytes || strings.TrimSpace(value) != value {
		return false
	}
	for _, character := range value {
		if character >= 'a' && character <= 'z' || character >= '0' && character <= '9' ||
			character == '.' || character == '-' || character == '_' {
			continue
		}
		return false
	}
	return true
}

func runContent(run Run) ([]byte, error) {
	content, err := json.Marshal(runBody{
		Version: run.Version, Recipe: run.Recipe, Outcome: run.Outcome,
		Inputs: run.Inputs, Outputs: run.Outputs, Failure: run.Failure,
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

func sameRun(left, right Run) bool {
	return left.Version == right.Version && left.Recipe == right.Recipe &&
		left.Outcome == right.Outcome && left.Failure == right.Failure &&
		slices.Equal(left.Inputs, right.Inputs) && slices.Equal(left.Outputs, right.Outputs)
}

func sameEvaluation(left, right Evaluation) bool {
	return left.Version == right.Version && left.Recipe == right.Recipe &&
		left.Run == right.Run && left.Dataset == right.Dataset && slices.Equal(left.Metrics, right.Metrics)
}
