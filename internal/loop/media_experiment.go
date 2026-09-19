package loop

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strconv"

	"overgo/internal/artifact"
)

// A media experiment acquires one bounded acquisition-to-report slice: it binds
// its model and data identities, the per-cell requests, and its budget as one
// typed protocol, then resolves every retained acquisition before opening a
// model. An identical rerun reuses the retained cells with no model load and no
// mutation; an interrupted run recovers only the missing cells. Storage rides
// the existing artifact owner and the media output stays with its compatibility
// owner; this is one slice, not a general experiment framework.
const (
	mediaExperimentMedia  = "application/vnd.overgo.media-experiment+json"
	mediaExperimentSchema = "overgo/media-experiment/v1"

	mediaAcquisitionMedia  = "application/vnd.overgo.media-acquisition+json"
	mediaAcquisitionSchema = "overgo/media-acquisition/v1"

	mediaReportMedia  = "application/vnd.overgo.media-experiment-report+json"
	mediaReportSchema = "overgo/media-experiment-report/v1"
)

// MediaCell is one acquisition request in a media experiment.
type MediaCell struct {
	Index   int         `json:"index"`
	Request artifact.ID `json:"request"`
}

// MediaExperiment is the typed acquisition protocol. Its identity is the content
// hash of its bound model, data, budget, and cells, so an identical experiment
// resolves the same retained acquisitions.
type MediaExperiment struct {
	ID     artifact.ID `json:"-"`
	Model  artifact.ID `json:"model"`
	Data   artifact.ID `json:"data"`
	Budget artifact.ID `json:"budget"`
	Cells  []MediaCell `json:"cells"`
}

// MediaAcquisition is one completed cell: the produced output bound to its
// request, with the compute it spent.
type MediaAcquisition struct {
	ID         artifact.ID `json:"-"`
	Experiment artifact.ID `json:"experiment"`
	Index      int         `json:"index"`
	Request    artifact.ID `json:"request"`
	Output     artifact.ID `json:"output"`
	ComputeNS  uint64      `json:"compute_ns"`
}

// MediaExperimentReport aggregates the acquisitions in cell order. Its identity
// is the experiment and its acquisitions, so a rerun that reuses every retained
// cell produces the same report. Reused and Acquired are run-local accounting of
// how many cells were reused versus freshly acquired, not part of the identity.
type MediaExperimentReport struct {
	ID           artifact.ID   `json:"-"`
	Experiment   artifact.ID   `json:"experiment"`
	Acquisitions []artifact.ID `json:"acquisitions"`
	Reused       int           `json:"-"`
	Acquired     int           `json:"-"`
}

// Acquirer opens the model and produces one cell's output. It is the model-open
// boundary: ResumeMediaExperiment calls it only for cells without retained
// evidence, so a fully retained experiment opens no model.
type Acquirer func(context.Context, MediaCell) (output artifact.ID, computeNS uint64, err error)

var (
	mediaExperimentCodec = artifact.JSONDocumentCodec(
		"media experiment", artifact.KindEvidence, mediaExperimentMedia, mediaExperimentSchema,
		canonicalizeMediaExperiment,
		func(v MediaExperiment) artifact.ID { return v.ID },
		func(v *MediaExperiment, id artifact.ID) { v.ID = id },
		func(v MediaExperiment) MediaExperiment { v.Cells = slices.Clone(v.Cells); return v },
	)
	mediaAcquisitionCodec = artifact.JSONDocumentCodec(
		"media acquisition", artifact.KindEvidence, mediaAcquisitionMedia, mediaAcquisitionSchema,
		canonicalizeMediaAcquisition,
		func(v MediaAcquisition) artifact.ID { return v.ID },
		func(v *MediaAcquisition, id artifact.ID) { v.ID = id }, nil,
	)
	mediaReportCodec = artifact.JSONDocumentCodec(
		"media experiment report", artifact.KindEvidence, mediaReportMedia, mediaReportSchema,
		canonicalizeMediaReport,
		func(v MediaExperimentReport) artifact.ID { return v.ID },
		func(v *MediaExperimentReport, id artifact.ID) { v.ID = id },
		func(v MediaExperimentReport) MediaExperimentReport {
			v.Acquisitions = slices.Clone(v.Acquisitions)
			return v
		},
	)
)

// NewMediaExperiment computes the experiment identity from its bound content.
func NewMediaExperiment(model, data, budget artifact.ID, cells []MediaCell) (MediaExperiment, error) {
	return mediaExperimentCodec.New(MediaExperiment{Model: model, Data: data, Budget: budget, Cells: cells})
}

func canonicalizeMediaExperiment(v *MediaExperiment) error {
	if v == nil || !v.Model.Valid() || !v.Data.Valid() || !v.Budget.Valid() || len(v.Cells) == 0 {
		return errors.New("loop: media experiment requires model, data, budget, and at least one cell")
	}
	seen := make(map[int]bool, len(v.Cells))
	for index, cell := range v.Cells {
		if cell.Index != index || seen[cell.Index] || !cell.Request.Valid() {
			return errors.New("loop: media experiment cells must be dense, ordered, and each bind a request")
		}
		seen[cell.Index] = true
	}
	return nil
}

func canonicalizeMediaAcquisition(v *MediaAcquisition) error {
	if v == nil || !v.Experiment.Valid() || v.Index < 0 || !v.Request.Valid() || !v.Output.Valid() {
		return errors.New("loop: media acquisition requires experiment, index, request, and output")
	}
	return nil
}

func canonicalizeMediaReport(v *MediaExperimentReport) error {
	if v == nil || !v.Experiment.Valid() || len(v.Acquisitions) == 0 {
		return errors.New("loop: media experiment report requires an experiment and its acquisitions")
	}
	for _, id := range v.Acquisitions {
		if !id.Valid() {
			return errors.New("loop: media experiment report has an invalid acquisition")
		}
	}
	return nil
}

func mediaAcquisitionAlias(experiment artifact.ID, index int) string {
	return "media/acquisitions/" + experiment.String() + "/" + strconv.Itoa(index)
}

func mediaReportAlias(experiment artifact.ID) string {
	return "media/experiments/" + experiment.String()
}

// ResumeMediaExperiment discovers every retained acquisition before opening a
// model, acquires only the missing cells through acquire, checkpoints each one
// idempotently, and publishes an idempotent report. A fully retained experiment
// makes no acquire call; an interrupted one recovers only its missing cells; a
// second run mutates nothing and returns the same report.
func ResumeMediaExperiment(
	ctx context.Context,
	repository artifact.Repository,
	experiment MediaExperiment,
	acquire Acquirer,
) (MediaExperimentReport, error) {
	canonical, err := NewMediaExperiment(experiment.Model, experiment.Data, experiment.Budget, experiment.Cells)
	if err != nil {
		return MediaExperimentReport{}, err
	}
	if canonical.ID != experiment.ID {
		return MediaExperimentReport{}, errors.New("loop: media experiment identity does not match its bound content")
	}
	completed, err := loadMediaAcquisitions(ctx, repository, experiment)
	if err != nil {
		return MediaExperimentReport{}, err
	}
	acquisitions := make([]artifact.ID, len(experiment.Cells))
	reused, acquired := 0, 0
	for position, cell := range experiment.Cells {
		if retained, ok := completed[cell.Index]; ok {
			acquisitions[position] = retained
			reused++
			continue
		}
		if acquire == nil {
			return MediaExperimentReport{}, errors.New("loop: media experiment has missing cells but no acquirer")
		}
		output, computeNS, err := acquire(ctx, cell)
		if err != nil {
			return MediaExperimentReport{}, fmt.Errorf("loop: media acquisition cell %d: %w", cell.Index, err)
		}
		stored, err := publishMediaAcquisition(ctx, repository, experiment, MediaAcquisition{
			Experiment: experiment.ID, Index: cell.Index, Request: cell.Request, Output: output, ComputeNS: computeNS,
		})
		if err != nil {
			return MediaExperimentReport{}, err
		}
		acquisitions[position] = stored
		acquired++
	}
	report, err := mediaReportCodec.New(MediaExperimentReport{
		Experiment: experiment.ID, Acquisitions: acquisitions, Reused: reused, Acquired: acquired,
	})
	if err != nil {
		return MediaExperimentReport{}, err
	}
	return publishMediaReport(ctx, repository, experiment, report)
}

// loadMediaAcquisitions resolves every retained cell before a model is opened,
// rejecting a retained acquisition that does not match its experiment and cell.
func loadMediaAcquisitions(
	ctx context.Context,
	repository artifact.Repository,
	experiment MediaExperiment,
) (map[int]artifact.ID, error) {
	completed := make(map[int]artifact.ID, len(experiment.Cells))
	for _, cell := range experiment.Cells {
		target, ok, err := artifact.ResolveAlias(ctx, repository, mediaAcquisitionAlias(experiment.ID, cell.Index))
		if err != nil {
			return nil, err
		}
		if !ok {
			continue
		}
		content, found, err := artifact.ReadContent(ctx, repository, target)
		if err != nil {
			return nil, err
		}
		if !found {
			return nil, errors.New("loop: retained media acquisition content is absent")
		}
		acquisition, err := mediaAcquisitionCodec.Parse(content.Data)
		if err != nil {
			return nil, err
		}
		if acquisition.ID != target || acquisition.Experiment != experiment.ID || acquisition.Index != cell.Index || acquisition.Request != cell.Request {
			return nil, fmt.Errorf("loop: retained media acquisition for cell %d does not match the experiment", cell.Index)
		}
		completed[cell.Index] = target
	}
	return completed, nil
}

func publishMediaAcquisition(
	ctx context.Context,
	repository artifact.Repository,
	experiment MediaExperiment,
	acquisition MediaAcquisition,
) (artifact.ID, error) {
	stored, err := mediaAcquisitionCodec.New(acquisition)
	if err != nil {
		return artifact.ID{}, err
	}
	content, err := mediaAcquisitionCodec.Content(stored)
	if err != nil {
		return artifact.ID{}, err
	}
	alias := mediaAcquisitionAlias(experiment.ID, acquisition.Index)
	lineage := artifact.DependencyLineage(stored.ID, acquisition.Request, acquisition.Output)
	batch, err := artifact.NewDocumentBatch(
		alias, []artifact.Content{content}, lineage,
		[]artifact.AliasBinding{{Name: alias, Target: stored.ID}},
	)
	if err != nil {
		return artifact.ID{}, err
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return artifact.ID{}, err
	}
	return stored.ID, nil
}

// publishMediaReport publishes the report once and verifies, never rewrites, a
// retained report, so a second run is a no-op.
func publishMediaReport(
	ctx context.Context,
	repository artifact.Repository,
	experiment MediaExperiment,
	report MediaExperimentReport,
) (MediaExperimentReport, error) {
	alias := mediaReportAlias(experiment.ID)
	current, exists, err := artifact.ResolveAlias(ctx, repository, alias)
	if err != nil {
		return MediaExperimentReport{}, err
	}
	if exists {
		if current != report.ID {
			return MediaExperimentReport{}, errors.New("loop: media experiment report differs from the retained report")
		}
		content, found, err := artifact.ReadContent(ctx, repository, current)
		if err != nil {
			return MediaExperimentReport{}, err
		}
		if !found {
			return MediaExperimentReport{}, errors.New("loop: retained media experiment report content is absent")
		}
		if _, err := mediaReportCodec.Parse(content.Data); err != nil {
			return MediaExperimentReport{}, err
		}
		return report, nil
	}
	content, err := mediaReportCodec.Content(report)
	if err != nil {
		return MediaExperimentReport{}, err
	}
	batch, err := artifact.NewDocumentBatch(
		alias, []artifact.Content{content},
		artifact.DependencyLineage(report.ID, report.Acquisitions...),
		[]artifact.AliasBinding{{Name: alias, Target: report.ID}},
	)
	if err != nil {
		return MediaExperimentReport{}, err
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return MediaExperimentReport{}, err
	}
	return report, nil
}
