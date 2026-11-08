package evaluation

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"overgo/internal/artifact"
)

const (
	// recordedEvidence opens the evidence an accepted exact verification records, before the plan it ran.
	recordedEvidence = "contract=exact;plan="
	// recordedSchema is the schema of a suite read back from the store, whose cases are those its verification ran.
	recordedSchema = "overgo/recorded-exact/v1"
)

// RecordedExactSuite is the exact suite an accepted verification ran, read back from the store: the evidence names the
// plan it ran, the plan its dataset, and the dataset holds the cases, so the suite read back names the same dataset
// (CompileExact identifies a dataset by its cases). On 2026-11-08 five servable models, the judge among them, held
// evidence for an architecture profile their build no longer compiled, and nothing could ask them their verification
// again.
func RecordedExactSuite(ctx context.Context, store artifact.Reader, evidence string) (ExactSuite, error) {
	named, exact := strings.CutPrefix(evidence, recordedEvidence)
	if !exact {
		return ExactSuite{}, fmt.Errorf("evaluation: the verification %q ran no exact suite", evidence)
	}
	named, _, _ = strings.Cut(named, ";")
	planID, err := artifact.ParseID(named)
	if err != nil {
		return ExactSuite{}, err
	}
	planContent, found, err := artifact.ReadContent(ctx, store, planID)
	if err != nil || !found {
		return ExactSuite{}, errors.Join(err, fmt.Errorf("evaluation: the store holds no plan %s", planID))
	}
	plan, err := ParsePlan(planContent.Data)
	if err != nil {
		return ExactSuite{}, err
	}
	datasetContent, found, err := artifact.ReadContent(ctx, store, plan.Dataset())
	if err != nil || !found {
		return ExactSuite{}, errors.Join(err, fmt.Errorf("evaluation: the store holds no cases of %s", plan.Dataset()))
	}
	suite := ExactSuite{Schema: recordedSchema, Source: plan.Dataset().String()}
	if err := json.Unmarshal(datasetContent.Data, &suite.Cases); err != nil {
		return ExactSuite{}, fmt.Errorf("evaluation: the cases of %s: %w", plan.Dataset(), err)
	}
	return suite, nil
}
