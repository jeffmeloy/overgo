package evaluation

import (
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

func (campaign *Campaign) resourceFitness(workload artifact.ID, run runrecord.Run) (runrecord.ResourceFitness, error) {
	if campaign == nil || run.Version != artifact.SecondDocumentVersion ||
		campaign.identity.Model.Kind() != artifact.KindModel ||
		run.Recipe != campaign.identity.Recipe || run.Environment != campaign.environment.ID ||
		workload.Kind() != artifact.KindProfile || !slices.Contains(run.Inputs, workload) {
		return runrecord.ResourceFitness{}, errors.New("evaluation: run resource authorities differ")
	}
	if err := run.ValidateIdentity(); err != nil {
		return runrecord.ResourceFitness{}, err
	}
	return runrecord.NewResourceFitness(runrecord.ResourceFitness{
		Scope: runrecord.ResourceScope{
			Surface: runrecord.SurfaceEvaluation,
			Model:   campaign.identity.Model, Hardware: run.Environment,
			Workload: workload, Attempt: run.ID,
		},
		Measures: []runrecord.ResourceMeasure{{Metric: runrecord.ResourceWallNS, Value: run.MeasuredNS}},
	})
}
