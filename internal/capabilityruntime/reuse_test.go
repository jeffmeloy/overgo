package capabilityruntime

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipetest"
	"overgo/internal/recipe"
	"overgo/internal/workflowruntime"
)

// reusedModel is a capability's model of this package, so what a build counts
// for this package is the identity its answers are kept under.
type reusedModel struct{}

// scoreRequest is what a scorer is asked: the observation it scores and the
// protocol it scores it under.
type scoreRequest struct {
	Observation int    `json:"observation"`
	Protocol    string `json:"protocol"`
}

// TestStoredResultIsReusedOnlyUnderItsOwnIdentity serves one request and
// scores its answer through a row of builds over one store, and counts what
// each build executed and scored. The model is a capability of one package and
// the scorer a capability of another, whose request names the observation and
// the protocol, so an answer is kept under the model's identity and the exact
// request, and a score under its observation, its scorer and its protocol.
//
// A build that changed nothing either depends on, though its commit is
// another, executes nothing and scores nothing. A change to the scorer alone
// scores again and executes nothing. A change to a kernel of the model
// executes again, and scores nothing while the answer is the one already
// scored; a change to a file the model reads beside its inventory executes
// again; a mended model executes again and its new answer is scored. A
// request for a fresh measurement executes every time, under an identity that
// holds an answer. And a build that counted no identity gives again nothing a
// process before it answered.
func TestStoredResultIsReusedOnlyUnderItsOwnIdentity(t *testing.T) {
	// Not parallel: the package's identities and code scope are this test's to move, and the tests that run in parallel wait for it.
	counted, built := execution, codeScope
	defer func() { execution, codeScope = counted, built }()
	const modelPackage, scorerPackage = "overgo/internal/capabilityruntime", "strings"
	identities := map[string]string{}
	execution = func(pkg string) (string, bool) {
		identity, held := identities[pkg]
		return identity, held
	}

	store, _, program := capabilityFixture(t, "reused")
	selection := modelrecipetest.CandidateExecution(t, store, program)
	preprocessor := filepath.Join(t.TempDir(), "tokenizer.json")
	if err := os.WriteFile(preprocessor, []byte("vocabulary"), 0o644); err != nil {
		t.Fatal(err)
	}
	executed, scored, gain := 0, 0, 2
	contract := artifact.JSONContract(artifact.KindOutput, "test.reuse-output.v1")
	model := Reading(JSONScalar[scalarRequest, reusedModel, int](
		"scalar", func(scalarRequest) error { return nil },
		func(context.Context, artifact.Repository, string, recipe.Program, scalarRequest) (reusedModel, error) {
			return reusedModel{}, nil
		},
		func(runtime *workflowruntime.Runtime, bound artifact.ID, _ reusedModel) error {
			return workflowruntime.RegisterJSONStage[scalarRequest, int](runtime, scalarModule, bound, contract, func(request scalarRequest) (int, error) {
				executed++
				return request.Value * gain, nil
			})
		},
	), func(string) []string { return []string{preprocessor} })
	scorer := JSONScalar[scoreRequest, *strings.Builder, int](
		"scalar", func(scoreRequest) error { return nil },
		func(context.Context, artifact.Repository, string, recipe.Program, scoreRequest) (*strings.Builder, error) {
			return &strings.Builder{}, nil
		},
		func(runtime *workflowruntime.Runtime, bound artifact.ID, _ *strings.Builder) error {
			return workflowruntime.RegisterJSONStage[scoreRequest, int](runtime, scalarModule, bound, contract, func(request scoreRequest) (int, error) {
				scored++
				return request.Observation + len(request.Protocol), nil
			})
		},
	)
	fresh := false
	measure := func() (answer, score int) {
		t.Helper()
		ctx := t.Context()
		if fresh {
			ctx = FreshRun(ctx)
		}
		got, err := model(ctx, store, "model", selection, `{"value":4}`)
		if err != nil {
			t.Fatal(err)
		}
		answer = got.(Measured).Output.(int)
		got, err = scorer(t.Context(), store, "scorer", selection, `{"observation":`+strconv.Itoa(answer)+`,"protocol":"exact"}`)
		if err != nil {
			t.Fatal(err)
		}
		return answer, got.(Measured).Output.(int)
	}
	for _, build := range []struct {
		what             string
		change           func()
		executed, scored int
		answer           int
	}{
		{"the first build", func() {
			identities[modelPackage], identities[scorerPackage] = "model-1", "scorer-1"
			codeScope = func() string { return "commit/first" }
		}, 1, 1, 8},
		{"a commit that changes nothing either depends on", func() {
			identities["overgo/internal/trainingworkflow"] = "training-2"
			codeScope = func() string { return "commit/second" }
		}, 1, 1, 8},
		{"a change to the scorer alone", func() { identities[scorerPackage] = "scorer-2" }, 1, 2, 8},
		{"a change to a kernel of the model, the answer the same", func() { identities[modelPackage] = "model-2" }, 2, 2, 8},
		{"a change to a file the model reads beside its inventory", func() {
			if err := os.WriteFile(preprocessor, []byte("a longer vocabulary"), 0o644); err != nil {
				t.Fatal(err)
			}
		}, 3, 2, 8},
		{"a mended model", func() { identities[modelPackage], gain = "model-3", 3 }, 4, 3, 12},
		{"the same build again", func() {}, 4, 3, 12},
		{"a request for a fresh measurement", func() { fresh = true }, 5, 3, 12},
		{"a second request for a fresh measurement", func() {}, 6, 3, 12},
		{"the answer kept before the measurements", func() { fresh = false }, 6, 3, 12},
		{"a build that counted no identity", func() {
			clear(identities)
			codeScope = func() string { return "process/one" }
		}, 7, 4, 12},
		{"a second process of such a build", func() { codeScope = func() string { return "process/two" } }, 8, 5, 12},
	} {
		build.change()
		answer, score := measure()
		if executed != build.executed || scored != build.scored || answer != build.answer || score != answer+len("exact") {
			t.Fatalf("%s: answer %d scored %d after %d executions and %d scorings; want answer %d scored %d after %d and %d",
				build.what, answer, score, executed, scored, build.answer, build.answer+len("exact"), build.executed, build.scored)
		}
	}
}
