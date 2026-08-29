package runrecord

import (
	"bytes"
	"testing"

	"overgo/internal/executionfailure"
	"overgo/internal/overgodb"
)

const failureFixtureObservedNS = int64(1_700_000_000_000_000_000)

// TestFailureNormalizationCorpus pins the record layer over the
// classifier corpus: raw observations publish immutably, derived
// normalizations name their classifier version, and reclassification
// under a newer classifier lands a sibling record while the raw bytes
// stay identical.
func TestFailureNormalizationCorpus(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()

	corpus := []struct {
		name  string
		value FailureObservation
		cause executionfailure.Cause
	}{
		{"transport", FailureObservation{
			Source: "swap-proxy", Message: "dial tcp 127.0.0.1:8080: connection refused",
			ExitCode: 1, ObservedUnixNS: failureFixtureObservedNS,
		}, executionfailure.CauseNetwork},
		{"context budget", FailureObservation{
			Source: "serving-lane", Message: "request exceeds the maximum context length",
			ObservedUnixNS: failureFixtureObservedNS,
		}, executionfailure.CauseContextExhaustion},
		{"unlearned", FailureObservation{
			Source: "gate-check", Message: "the operation ended unexpectedly",
			Detail: "no further output", ObservedUnixNS: failureFixtureObservedNS,
		}, executionfailure.CauseUnknown},
	}
	for _, sample := range corpus {
		observation, err := PublishFailureObservation(ctx, store, sample.value)
		if err != nil {
			t.Fatalf("%s: publish observation: %v", sample.name, err)
		}
		before, err := observation.Content()
		if err != nil {
			t.Fatal(err)
		}
		derived := NormalizeFailureObservation(observation)
		if derived.Cause != sample.cause || derived.ClassifierVersion != executionfailure.ClassifierVersion {
			t.Fatalf("%s: derived classification = %+v", sample.name, derived)
		}
		published, err := PublishFailureNormalization(ctx, store, derived)
		if err != nil {
			t.Fatalf("%s: publish normalization: %v", sample.name, err)
		}

		// A newer classifier that learned this evidence publishes a
		// sibling under its own version; both classifications remain.
		newer := derived
		newer.ClassifierVersion = executionfailure.ClassifierVersion + 1
		newer.Cause = executionfailure.CauseProcessFailure
		newer.Rule = "learned-pattern"
		reclassified, err := PublishFailureNormalization(ctx, store, newer)
		if err != nil {
			t.Fatalf("%s: publish reclassification: %v", sample.name, err)
		}
		if reclassified.ID == published.ID {
			t.Fatalf("%s: reclassification reused the original record", sample.name)
		}
		for version, want := range map[uint16]FailureNormalization{
			derived.ClassifierVersion: published, newer.ClassifierVersion: reclassified,
		} {
			resolved, found, err := ResolveFailureNormalization(ctx, store, observation.ID, version)
			if err != nil || !found || resolved.ID != want.ID || resolved.Cause != want.Cause {
				t.Fatalf("%s: classifier v%d resolution = %+v found=%v err=%v", sample.name, version, resolved, found, err)
			}
		}

		// The raw observation is untouched by both classifications.
		reread, err := RequireFailureObservation(ctx, store, observation.ID)
		if err != nil {
			t.Fatal(err)
		}
		after, err := reread.Content()
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(before.Data, after.Data) {
			t.Fatalf("%s: raw evidence changed under reclassification", sample.name)
		}
	}

	refusals := []struct {
		name  string
		value FailureNormalization
	}{
		{"foreign cause", FailureNormalization{ClassifierVersion: 1, Cause: "crashed", Rule: "unmatched"}},
		{"zero classifier", FailureNormalization{Cause: executionfailure.CauseUnknown, Rule: "unmatched"}},
	}
	anchor, err := PublishFailureObservation(ctx, store, corpus[0].value)
	if err != nil {
		t.Fatal(err)
	}
	for _, refusal := range refusals {
		refusal.value.Observation = anchor.ID
		if _, err := PublishFailureNormalization(ctx, store, refusal.value); err == nil {
			t.Fatalf("%s: accepted", refusal.name)
		}
	}
	if _, err := PublishFailureObservation(ctx, store, FailureObservation{
		Source: "gate-check", ObservedUnixNS: failureFixtureObservedNS,
	}); err == nil {
		t.Fatal("observation without a message accepted")
	}
}
