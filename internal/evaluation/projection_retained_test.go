package evaluation

import (
	"encoding/json"
	"fmt"
	"maps"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

type nativeProjectionResult struct {
	Name         string
	Text         string
	PromptTokens int `json:"prompt_tokens"`
}

// Reconstruct acceptance from the original authorities without opening a model.
func TestRetainedProjectionResults(t *testing.T) {
	root := testutil.RepoRoot(t)
	var fixtures []struct {
		Name     string      `json:"name"`
		Evidence artifact.ID `json:"evidence"`
		Suite    artifact.ID `json:"suite"`
	}
	if err := jsonfile.DecodeStrict(filepath.Join(root, "docs/verification/projection_publications.json"), &fixtures); err != nil {
		t.Fatal(err)
	}
	if len(fixtures) == 0 {
		t.Fatal("projection denominator is empty")
	}
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	for _, fixture := range fixtures {
		t.Run(fixture.Name, func(t *testing.T) {
			var retained struct {
				Plan, Report, Recipe, Gate, Run, Environment, Qualification artifact.ID
				ModelDefinition                                             artifact.ID `json:"model_definition"`
				Producer                                                    string
				Cases                                                       int
				Inputs                                                      map[string]artifact.ID
			}
			readRetainedEvidence(t, store, fixture.Evidence, &retained)
			if !slices.Contains(slices.Collect(maps.Values(retained.Inputs)), fixture.Suite) {
				t.Fatal("suite is not bound to acquisition")
			}
			readText := func(id artifact.ID) string {
				t.Helper()
				var envelope struct{ Text string }
				readRetainedEvidence(t, store, id, &envelope)
				return envelope.Text
			}
			var suite ExactSuite
			if err := json.Unmarshal([]byte(readText(fixture.Suite)), &suite); err != nil {
				t.Fatal(err)
			}
			exact, err := CompileExact(suite)
			if err != nil {
				t.Fatal(err)
			}
			var body planBody
			readRetainedEvidence(t, store, retained.Plan, &body)
			policy, err := ReadExecutionPolicy(t.Context(), store, body.Execution)
			if err != nil {
				t.Fatal(err)
			}
			plan, err := BindExact(exact, ExactAuthorities{ModelDefinition: retained.ModelDefinition, RuntimeRecipe: retained.Recipe, CodeCommit: retained.Producer, Environment: retained.Environment, Execution: policy})
			if err != nil || plan.Identity() != retained.Plan || len(suite.Cases) != retained.Cases {
				t.Fatalf("bound plan or denominator differs: %v", err)
			}
			shards, err := compileExactShards(exact, plan)
			if err != nil {
				t.Fatal(err)
			}
			reports, err := loadShardReports(t.Context(), store, plan, shards)
			if err != nil || len(reports) != len(shards) {
				t.Fatalf("missing exact results: %v", err)
			}
			ordered := make([]shardReport, len(shards))
			for i, c := range suite.Cases {
				r := reports[i]
				if len(r.Observations) != 1 || !r.Observations[0].Passed || r.Observations[0].Failure != "" {
					t.Fatalf("failed case %s", c.Name)
				}
				var output textOutput
				readRetainedEvidence(t, store, r.Observations[0].Output, &output)
				if output.Text != c.Text {
					t.Fatalf("changed case %s", c.Name)
				}
				ordered[i] = r
			}
			merged, err := mergeShardReports(ordered)
			if err != nil || merged.ID != retained.Report {
				t.Fatalf("report differs: %v", err)
			}
			proof, err := runrecord.VerifyGateRun(t.Context(), store, retained.Recipe, retained.Gate, retained.Run)
			if err != nil {
				t.Fatal(err)
			}
			want := fmt.Sprintf("contract=exact;plan=%s;report=%s;cases=%d", retained.Plan, retained.Report, len(suite.Cases))
			if proof.Run.CodeCommit != retained.Producer || proof.Run.Environment != retained.Environment || len(proof.Gate.Steps) != 1 || proof.Gate.Steps[0].Evidence != want || proof.Run.MeasuredNS == 0 {
				t.Fatal("execution authority differs")
			}
			var qualification struct {
				Proposal    artifact.ID
				Qualified   bool
				Denominator int
				Failures    []string
				Files       map[string]artifact.ID
			}
			readRetainedEvidence(t, store, retained.Qualification, &qualification)
			if !qualification.Qualified || len(qualification.Failures) != 0 || qualification.Denominator != len(suite.Cases) || !strings.Contains(suite.Source, qualification.Proposal.String()) {
				t.Fatal("native qualification differs")
			}
			var proposal struct {
				Protocol struct{ Model, Projector artifact.ID }
			}
			readRetainedEvidence(t, store, qualification.Proposal, &proposal)
			definition, err := recipe.RequireDefinition(t.Context(), store, retained.Recipe)
			if err != nil {
				t.Fatal(err)
			}
			for dependency, want := range map[recipe.DependencyRole]artifact.ID{recipe.DependencyModel: proposal.Protocol.Model, recipe.DependencyProjector: proposal.Protocol.Projector} {
				got, found := definition.PrimaryDependency(dependency)
				if !found || got != want {
					t.Fatal("native and candidate artifacts differ")
				}
			}
			captures := 0
			for path, id := range qualification.Files {
				if !strings.HasSuffix(path, "/results.jsonl") {
					continue
				}
				captures++
				rows := map[string]nativeProjectionResult{}
				for line := range strings.SplitSeq(strings.TrimSpace(readText(id)), "\n") {
					var row nativeProjectionResult
					if err := json.Unmarshal([]byte(line), &row); err != nil {
						t.Fatal(err)
					}
					if _, ok := rows[row.Name]; ok {
						t.Fatal("duplicate native case")
					}
					rows[row.Name] = row
				}
				delete(rows, "legacy-control")
				if len(rows) != len(suite.Cases) {
					t.Fatal("native case denominator differs")
				}
				for _, c := range suite.Cases {
					row, ok := rows[c.Name]
					if !ok || row.Text != c.Text || row.PromptTokens != c.PromptTokens {
						t.Fatalf("native case differs: %s", c.Name)
					}
				}
			}
			if captures != 2 {
				t.Fatal("both native backends are required")
			}
			t.Logf("%d exact cases; native CPU/CUDA, producer, recipe, run and complete shards retained; model executions=0", len(suite.Cases))
		})
	}
}
