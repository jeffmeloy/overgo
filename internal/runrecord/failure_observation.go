package runrecord

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/executionfailure"
)

const (
	// FailureObservationMediaType identifies raw failure evidence records.
	FailureObservationMediaType = "application/vnd.overgo.failure-observation+json"
	// FailureObservationSchema is the observation document version.
	FailureObservationSchema = "overgo/failure-observation/v1"

	// FailureNormalizationMediaType identifies derived classifications.
	FailureNormalizationMediaType = "application/vnd.overgo.failure-normalization+json"
	// FailureNormalizationSchema is the normalization document version.
	FailureNormalizationSchema = "overgo/failure-normalization/v1"

	// FailureNormalizationAliasRoot scopes the current classification of
	// one observation under one classifier version.
	FailureNormalizationAliasRoot = "failure/normalization/"
)

var failureObservationCodec = artifact.JSONDocumentCodec(
	"failure observation", artifact.KindEvidence, FailureObservationMediaType, FailureObservationSchema,
	canonicalizeFailureObservation,
	func(value FailureObservation) artifact.ID { return value.ID },
	func(value *FailureObservation, id artifact.ID) { value.ID = id },
	func(value FailureObservation) FailureObservation { return value },
)

var failureNormalizationCodec = artifact.JSONDocumentCodec(
	"failure normalization", artifact.KindEvidence, FailureNormalizationMediaType, FailureNormalizationSchema,
	canonicalizeFailureNormalization,
	func(value FailureNormalization) artifact.ID { return value.ID },
	func(value *FailureNormalization, id artifact.ID) { value.ID = id },
	func(value FailureNormalization) FailureNormalization { return value },
)

// FailureObservation preserves one failure's raw evidence exactly as
// the failing component reported it. It is immutable once published;
// classifiers derive normalizations from it and never rewrite it.
type FailureObservation struct {
	Version uint16 `json:"version"`
	// Source labels the reporting component, such as a lane or a tool.
	Source  string `json:"source"`
	Message string `json:"message"`
	// Detail carries secondary raw evidence, typically a stderr tail.
	Detail         string      `json:"detail,omitempty"`
	ExitCode       int32       `json:"exit_code,omitempty"`
	Interrupted    bool        `json:"interrupted,omitempty"`
	ObservedUnixNS int64       `json:"observed_unix_ns"`
	Run            artifact.ID `json:"run,omitzero"`
	ID             artifact.ID `json:"-"`
}

// FailureNormalization is one derived classification of an observation
// onto the canonical cause vocabulary. It names the classifier version
// and rule that decided, so a newer classifier publishes a sibling
// record instead of replacing this one.
type FailureNormalization struct {
	Version           uint16                 `json:"version"`
	Observation       artifact.ID            `json:"observation"`
	ClassifierVersion uint16                 `json:"classifier_version"`
	Cause             executionfailure.Cause `json:"cause"`
	Rule              string                 `json:"rule"`
	ID                artifact.ID            `json:"-"`
}

// ParseFailureObservation returns a validated observation from bytes.
func ParseFailureObservation(content []byte) (FailureObservation, error) {
	return failureObservationCodec.Parse(content)
}

// RequireFailureObservation returns a validated repository observation.
func RequireFailureObservation(ctx context.Context, reader artifact.Reader, id artifact.ID) (FailureObservation, error) {
	return failureObservationCodec.Require(ctx, reader, id)
}

// Content returns the observation's canonical bytes.
func (value FailureObservation) Content() (artifact.Content, error) {
	return failureObservationCodec.Content(value)
}

// commitFailureDocument publishes one identified failure record with
// the edges its publisher computed; both failure publishers share this
// single commit path.
func commitFailureDocument[T any](
	ctx context.Context, repository artifact.Repository, codec artifact.DocumentCodec[T],
	key string, identified T, lineage []artifact.Lineage, aliases []artifact.AliasBinding,
) (T, error) {
	batch, err := codec.Batch(key, identified, lineage, aliases)
	if err != nil {
		var zero T
		return zero, err
	}
	_, err = artifact.CommitBatch(ctx, repository, batch)
	if err != nil {
		var zero T
		return zero, err
	}
	return identified, nil
}

// PublishFailureObservation identifies and commits one raw failure
// fact, bound to its run when the caller knows it.
func PublishFailureObservation(ctx context.Context, repository artifact.Repository, value FailureObservation) (FailureObservation, error) {
	if ctx == nil || repository == nil {
		return FailureObservation{}, errors.New("run record: failure observation repository is absent")
	}
	value.Version, value.ID = artifact.InitialDocumentVersion, artifact.ID{}
	identified, err := failureObservationCodec.New(value)
	if err != nil {
		return FailureObservation{}, err
	}
	var parents []artifact.ID
	if identified.Run.Valid() {
		parents = append(parents, identified.Run)
	}
	return commitFailureDocument(ctx, repository, failureObservationCodec,
		"failure/observation/"+identified.ID.String(), identified,
		artifact.DependencyLineage(identified.ID, parents...), nil)
}

// Evidence adapts the raw observation for the deterministic classifier.
func (value FailureObservation) Evidence() executionfailure.Evidence {
	return executionfailure.Evidence{Message: value.Message, Detail: value.Detail, ExitCode: value.ExitCode}
}

// NormalizeFailureObservation derives the current classifier's
// normalization of one published observation.
func NormalizeFailureObservation(observation FailureObservation) FailureNormalization {
	classified := executionfailure.Normalize(observation.Evidence())
	return FailureNormalization{
		Observation:       observation.ID,
		ClassifierVersion: classified.ClassifierVersion,
		Cause:             classified.Cause,
		Rule:              classified.Rule,
	}
}

// RequireFailureNormalization returns a validated derived record.
func RequireFailureNormalization(ctx context.Context, reader artifact.Reader, id artifact.ID) (FailureNormalization, error) {
	return failureNormalizationCodec.Require(ctx, reader, id)
}

// PublishFailureNormalization commits one derived classification. The
// observation must already exist and is linked as lineage; the alias
// binds one classification per observation and classifier version, so
// reclassification lands beside prior versions instead of over them.
func PublishFailureNormalization(ctx context.Context, repository artifact.Repository, value FailureNormalization) (FailureNormalization, error) {
	if ctx == nil || repository == nil {
		return FailureNormalization{}, errors.New("run record: failure normalization repository is absent")
	}
	if _, err := RequireFailureObservation(ctx, repository, value.Observation); err != nil {
		return FailureNormalization{}, err
	}
	value.Version, value.ID = artifact.InitialDocumentVersion, artifact.ID{}
	identified, err := failureNormalizationCodec.New(value)
	if err != nil {
		return FailureNormalization{}, err
	}
	alias := failureNormalizationAlias(identified.Observation, identified.ClassifierVersion)
	return commitFailureDocument(ctx, repository, failureNormalizationCodec,
		"failure/normalization/"+identified.ID.String(), identified,
		artifact.DependencyLineage(identified.ID, identified.Observation),
		[]artifact.AliasBinding{{Name: alias, Target: identified.ID}})
}

// ResolveFailureNormalization returns one observation's classification
// under one classifier version, when that classifier has run.
func ResolveFailureNormalization(ctx context.Context, reader artifact.Reader, observation artifact.ID, classifierVersion uint16) (FailureNormalization, bool, error) {
	return failureNormalizationCodec.Resolve(ctx, reader, failureNormalizationAlias(observation, classifierVersion))
}

func failureNormalizationAlias(observation artifact.ID, classifierVersion uint16) string {
	return FailureNormalizationAliasRoot + observation.String() + "/" + fmt.Sprint(classifierVersion)
}

func canonicalizeFailureObservation(value *FailureObservation) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		!validLabel(value.Source) || value.Message == "" || value.ObservedUnixNS <= 0 {
		return errors.New("run record: invalid failure observation")
	}
	if value.Run.Valid() && value.Run.Kind() != artifact.KindRun {
		return errors.New("run record: invalid failure observation run")
	}
	return nil
}

func canonicalizeFailureNormalization(value *FailureNormalization) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.Observation.Kind() != artifact.KindEvidence || value.ClassifierVersion == 0 ||
		!executionfailure.ValidCause(value.Cause) || !validLabel(value.Rule) {
		return errors.New("run record: invalid failure normalization")
	}
	return nil
}
