package gate

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/repoanalysis"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestModernCensusExistingContent(t *testing.T) {
	root := modernMemoFixture(t)
	path := filepath.Join(root, "internal/example/value.go")
	if err := os.WriteFile(path, []byte("package example\nfunc Value() int { return 2 }\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	first := modernMemoContext(t, root)
	_, detail, err := first.computeModernGo(t.Context(), automationcheck.Invocation{})
	if err != nil {
		t.Fatal(err)
	}
	var before modernGoReferences
	if err := json.Unmarshal([]byte(detail), &before); err != nil || before.Candidate == before.Base {
		t.Fatalf("fixture must publish distinct source and base: %+v, %v", before, err)
	}
	input := first.modernInput.ID
	store, err := first.openStore()
	if err != nil {
		t.Fatal(err)
	}
	head, sequence := store.Head()
	if err := first.closeStore(); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, root, "add", "internal/example/value.go")
	runGitFixture(t, root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "advance source baseline")
	next := modernMemoContext(t, root)
	_, detail, err = next.computeModernGo(t.Context(), automationcheck.Invocation{})
	if err != nil {
		t.Fatal("already-published census must remain usable after HEAD advances:", err)
	}
	var after modernGoReferences
	if err := json.Unmarshal([]byte(detail), &after); err != nil || after.Candidate != before.Candidate || after.Base != after.Candidate || next.modernInput.ID == input {
		t.Fatalf("changed computation input lost exact payload identity: %+v, %v", after, err)
	}
	store, err = next.openStore()
	if err != nil {
		t.Fatal(err)
	}
	if gotHead, gotSequence := store.Head(); gotHead != head || gotSequence != sequence {
		t.Fatal("existing census content produced a redundant store commit")
	}
	if _, _, found, err := next.readModernGoOutput(automationcheck.Evidence{Detail: detail}); err != nil || !found {
		t.Fatalf("idempotent publication lost readable output: found=%t err=%v", found, err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if _, _, err := next.computeModernGo(t.Context(), automationcheck.Invocation{}); !errors.Is(err, overgodb.ErrClosed) {
		t.Fatalf("closed-store error was hidden: %v", err)
	}
}

func modernMemoFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	for _, path := range []string{"cmd/example", "internal/example", "internal/repoanalysis", "docs"} {
		if err := os.MkdirAll(filepath.Join(root, path), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	for name, text := range map[string]string{
		"go.mod":                    "module example\n\ngo 1.26\n",
		".gitignore":                "tmp/\novergodb-store/\nother-store/\n",
		"cmd/example/main.go":       "package main\nfunc main() {}\n",
		"internal/example/value.go": "package example\nfunc Value() int { return 1 }\n",
	} {
		if err := os.WriteFile(filepath.Join(root, name), []byte(text), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	modernMemoPublish(t, root)
	runGitFixture(t, root, "init")
	runGitFixture(t, root, "add", ".")
	runGitFixture(t, root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "baseline")
	return root
}

func modernMemoPublish(t *testing.T, root string) {
	t.Helper()
	census, err := repoanalysis.BuildModernGoCensus(root, repoanalysis.ModernGoTargetVersion)
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := repoanalysis.BuildModernGoBaseline(census)
	if err != nil {
		t.Fatal(err)
	}
	published, err := repoanalysis.BuildModernGoPublishedCensus(census, baseline)
	if err != nil {
		t.Fatal(err)
	}
	if err := jsonfile.Write(filepath.Join(root, repoanalysis.ModernGoBaselineFile), baseline, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := jsonfile.Write(filepath.Join(root, repoanalysis.ModernGoPublishedCensusFile), published, 0o644); err != nil {
		t.Fatal(err)
	}
}

func modernMemoContext(t *testing.T, root string) *gateContext {
	t.Helper()
	g := &gateContext{repo: root, storePath: "overgodb-store", environment: lifecycleTestEnvironment(t)}
	t.Cleanup(func() {
		if err := g.closeStore(); err != nil {
			t.Error(err)
		}
	})
	return g
}

// Execute the production computation and admission nodes. Scope admission is
// a fixture prerequisite; the full graph's ordering has its own executable test.
func runModernMemo(t *testing.T, g *gateContext, generation string, cache *automationcheck.EvidenceCache) map[string]automationcheck.DAGResult {
	t.Helper()
	var checks []automationcheck.Check
	for _, check := range g.pipelineChecks() {
		switch check.Descriptor.Name {
		case modernCensusCheckName:
			check.Descriptor.Dependencies = nil
			checks = append(checks, check)
		case "modern-go":
			checks = append(checks, check)
		}
	}
	invocations, err := automationcheck.Plan(checks, automationcheck.Impact{})
	if err != nil {
		t.Fatal(err)
	}
	input, err := g.prepareModernGoInput()
	if err != nil {
		t.Fatal(err)
	}
	tree := sha256.Sum256([]byte(generation))
	manifest, err := automationcheck.BindManifestPlan(
		testutil.ArtifactID(t, artifact.KindProfile, "base"), testutil.ArtifactID(t, artifact.KindProfile, "candidate"),
		input.Source.Identity(), fmt.Sprintf("%x", tree), automationcheck.Surface{Identity: "census fixture"}, automationcheck.Impact{}, invocations,
	)
	if err != nil {
		t.Fatal(err)
	}
	g.manifestPlan = &manifest
	inputs := map[artifact.ID]artifact.ID{}
	for index, invocation := range invocations {
		bound, err := g.bindCheckExecution(manifest, invocation, input.ID)
		if err != nil {
			t.Fatal(err)
		}
		invocations[index] = bound
		inputs[bound.ID] = input.ID
	}
	results, err := g.executeChecks(invocations, nil, inputs, cache, nil)
	if err != nil {
		t.Fatal(err)
	}
	byName := map[string]automationcheck.DAGResult{}
	for _, result := range results {
		byName[result.Invocation.Check.Name] = result
	}
	return byName
}

func testModernCensusRestartAndLiveAdmission(t *testing.T) {
	root := modernMemoFixture(t)
	first := modernMemoContext(t, root)
	cache := first.loadRetryCache()
	initial := runModernMemo(t, first, "first", &cache)
	for name, result := range initial {
		if result.Err != nil || result.Evidence.Inapplicable || result.Evidence.Reused {
			t.Fatalf("initial %s: %+v", name, result)
		}
	}
	original := initial[modernCensusCheckName].Evidence
	if err := first.closeStore(); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "docs", "plan.json"), []byte("{\"changed\":true}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	restart := modernMemoContext(t, root)
	cache = restart.loadRetryCache()
	warm := runModernMemo(t, restart, "successor plan", &cache)
	reused := warm[modernCensusCheckName].Evidence
	if !reused.Reused || reused.Source == nil || reused.Source.Evidence != original.ID || reused.ID == original.ID {
		t.Fatalf("restart lost original provenance: %+v", reused)
	}
	if err := automationcheck.ValidateReuseAuthority(reused, original.Authority.Definition); err != nil {
		t.Fatal(err)
	}
	if warm["modern-go"].Err != nil || warm["modern-go"].Evidence.Reused {
		t.Fatalf("live admission: %+v", warm["modern-go"])
	}
	if err := restart.closeStore(); err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"published report", "baseline source", "expired exception"} {
		t.Run(mode, func(t *testing.T) {
			modernMemoPublish(t, root)
			path := filepath.Join(root, repoanalysis.ModernGoBaselineFile)
			baseline, err := repoanalysis.LoadModernGoBaseline(path)
			if err != nil {
				t.Fatal(err)
			}
			switch mode {
			case "published report":
				path := filepath.Join(root, repoanalysis.ModernGoPublishedCensusFile)
				var published repoanalysis.ModernGoPublishedCensus
				if err := jsonfile.DecodeStrict(path, &published); err != nil {
					t.Fatal(err)
				}
				published.Adopted++
				if err := jsonfile.Write(path, published, 0o644); err != nil {
					t.Fatal(err)
				}
			case "baseline source":
				baseline.SourceIdentity = fmt.Sprintf("%x", sha256.Sum256([]byte("foreign source")))
			case "expired exception":
				baseline.Exceptions = []repoanalysis.ModernGoException{{
					Guideline: baseline.Guidelines[0].ID, Path: "internal/example/value.go", Symbol: "Value", Owner: "internal/example",
					Reason: "Fixture exercises live expiry", Oracle: "go test ./internal/example", Retirement: "Fixture only",
					Expires: time.Now().UTC().AddDate(-1, 0, 0).Format(time.DateOnly), CandidateCeiling: 1,
				}}
				baseline.ExceptionSHA256, err = repoanalysis.ModernGoExceptionIdentity(baseline.Exceptions)
				if err != nil {
					t.Fatal(err)
				}
			}
			if err := jsonfile.Write(path, baseline, 0o644); err != nil {
				t.Fatal(err)
			}
			policy := modernMemoContext(t, root)
			cache := policy.loadRetryCache()
			rejected := runModernMemo(t, policy, mode, &cache)
			if !rejected[modernCensusCheckName].Evidence.Reused || rejected["modern-go"].Err == nil {
				t.Fatalf("cached computation bypassed %s", mode)
			}
		})
	}
	modernMemoPublish(t, root)
	testModernMemoLiveAuthority(t, root)
}

func testModernCensusMissingAndUnprovenResults(t *testing.T) {
	root := modernMemoFixture(t)
	first := modernMemoContext(t, root)
	cache := first.loadRetryCache()
	initial := runModernMemo(t, first, "first", &cache)
	if err := initial["modern-go"].Err; err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(cache)
	if err != nil {
		t.Fatal(err)
	}
	if err := first.closeStore(); err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"missing receipt", "failed receipt", "skipped receipt", "unfinished receipt", "foreign input", "foreign environment", "missing output"} {
		t.Run(mode, func(t *testing.T) {
			g := modernMemoContext(t, root)
			var cache automationcheck.EvidenceCache
			if err := json.Unmarshal(encoded, &cache); err != nil {
				t.Fatal(err)
			}
			for key, entry := range cache.Entries {
				switch mode {
				case "missing receipt":
					delete(cache.Entries, key)
					continue
				case "failed receipt":
					entry.Outcome = runrecord.LaneFailed
				case "skipped receipt":
					skipped := automationcheck.Evidence{InvocationID: entry.Source.InvocationID, Authority: entry.Source.Authority, Outcome: runrecord.LanePassed, Inapplicable: true}
					entry.Source.Evidence, err = skipped.Identity()
					if err != nil {
						t.Fatal(err)
					}
				case "unfinished receipt":
					entry.Source = nil
				case "foreign input":
					entry.Input = testutil.ArtifactID(t, artifact.KindProfile, "foreign")
				case "foreign environment":
					entry.Source.Environment = testutil.ArtifactID(t, artifact.KindEvidence, "foreign environment")
				case "missing output":
					g.storePath = "other-store"
				}
				cache.Entries[key] = entry
			}
			result := runModernMemo(t, g, mode, &cache)
			if result[modernCensusCheckName].Evidence.Reused || result["modern-go"].Err != nil {
				t.Fatalf("result not repaired: %+v", result)
			}
		})
	}
}

func testModernCensusSourceAndBuildInvalidation(t *testing.T) {
	root := modernMemoFixture(t)
	first := modernMemoContext(t, root)
	before, err := first.prepareModernGoInput()
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"internal/example/value.go", "internal/repoanalysis/analyzer.go", "go.mod"} {
		path := filepath.Join(root, name)
		original, readErr := os.ReadFile(path)
		changed := append(slices.Clone(original), []byte("\n// changed input\n")...)
		if os.IsNotExist(readErr) {
			changed = []byte("package example\n")
		}
		if err := os.WriteFile(path, changed, 0o644); err != nil {
			t.Fatal(err)
		}
		g := modernMemoContext(t, root)
		after, err := g.prepareModernGoInput()
		if err != nil || after.ID == before.ID {
			t.Fatalf("%s did not invalidate: %v", name, err)
		}
		if os.IsNotExist(readErr) {
			if err := os.Remove(path); err != nil {
				t.Fatal(err)
			}
		} else if err := os.WriteFile(path, original, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	changed := modernMemoContext(t, root)
	environment := changed.environment
	environment.Runtime += ";changed toolchain"
	changed.environment, err = runrecord.NewEnvironment(environment)
	if err != nil {
		t.Fatal(err)
	}
	after, err := changed.prepareModernGoInput()
	if err != nil || after.ID == before.ID {
		t.Fatalf("toolchain did not invalidate: %v", err)
	}
	if !reflect.DeepEqual(before.Selection.Files, after.Selection.Files) {
		t.Fatal("environment test changed selected files")
	}
	t.Run("selected build files", func(t *testing.T) {
		path := filepath.Join(root, "internal/example/tagged.go")
		if err := os.WriteFile(path, []byte("//go:build census_fixture\n\npackage example\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		before, err := modernMemoContext(t, root).prepareModernGoInput()
		if err != nil {
			t.Fatal(err)
		}
		t.Setenv("GOFLAGS", strings.TrimSpace(os.Getenv("GOFLAGS")+" -tags=census_fixture"))
		after, err := modernMemoContext(t, root).prepareModernGoInput()
		if err != nil || after.ID == before.ID || reflect.DeepEqual(before.Selection.Files, after.Selection.Files) {
			t.Fatalf("build selection did not invalidate: %v", err)
		}
	})
	t.Run("platform", func(t *testing.T) {
		before, err := modernMemoContext(t, root).prepareModernGoInput()
		if err != nil {
			t.Fatal(err)
		}
		target := "linux"
		if runtime.GOOS == target {
			target = "windows"
		}
		t.Setenv("GOOS", target)
		after, err := modernMemoContext(t, root).prepareModernGoInput()
		if err != nil || after.ID == before.ID || after.Selection.Context == before.Selection.Context {
			t.Fatalf("platform did not invalidate: %v", err)
		}
	})
	t.Run("target Go", func(t *testing.T) {
		before, err := modernMemoContext(t, root).prepareModernGoInput()
		if err != nil {
			t.Fatal(err)
		}
		// The preceding supported language floor changes the applicable catalog.
		census, err := repoanalysis.BuildModernGoCensus(root, "1.25")
		if err != nil {
			t.Fatal(err)
		}
		baseline, err := repoanalysis.BuildModernGoBaseline(census)
		if err != nil {
			t.Fatal(err)
		}
		if err := jsonfile.Write(filepath.Join(root, repoanalysis.ModernGoBaselineFile), baseline, 0o644); err != nil {
			t.Fatal(err)
		}
		after, err := modernMemoContext(t, root).prepareModernGoInput()
		if err != nil || after.ID == before.ID || after.TargetGo == before.TargetGo || after.Source.Identity() != before.Source.Identity() {
			t.Fatalf("target Go did not invalidate: %v", err)
		}
	})
}

func testModernMemoLiveAuthority(t *testing.T, root string) {
	t.Helper()
	t.Setenv(plan.AutomationWorkerEnvironment, "")
	t.Setenv(plan.AutomationRoleEnvironment, "fixture")
	path := filepath.Join(root, plan.Path)
	document := []byte(`{"campaign":"fixture","doctrine":"fixture","items":[{"id":"work","owner":"fixture","status":"open","steps":[{"id":"do","status":"open","verify":"go test ./internal/example"}]}]}`)
	if err := os.WriteFile(path, document, 0o644); err != nil {
		t.Fatal(err)
	}
	g := modernMemoContext(t, root)
	cache := g.loadRetryCache()
	results := runModernMemo(t, g, "current owned plan", &cache)
	if !results[modernCensusCheckName].Evidence.Reused || results["modern-go"].Err != nil {
		t.Fatal("plan changed the pure computation")
	}
	if err := checkPlanBindingForTest(root, "work/do"); err != nil {
		t.Fatal(err)
	}
	if err := checkPlanBindingForTest(root, "work/absent"); err == nil {
		t.Fatal("warm census admitted an absent step")
	}
	t.Setenv(plan.AutomationRoleEnvironment, "foreign")
	if err := checkPlanBindingForTest(root, "work/do"); err == nil {
		t.Fatal("warm census admitted a foreign role")
	}
	g.paths = []string{plan.Path}
	planned, err := g.treeStateKey()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(document, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := g.requireCandidateTree(planned); err == nil {
		t.Fatal("warm census admitted a changed candidate")
	}
	g.indexBefore, err = captureGateIndex(root)
	if err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, root, "add", "--", plan.Path)
	if err := g.requireGateStartState(); err == nil || !strings.Contains(err.Error(), "index moved") {
		t.Fatalf("warm census index admission: %v", err)
	}
	if phaseReusesEvidence("modern-go") || phaseReusesEvidence("commit") {
		t.Fatal("live admission was made cacheable")
	}
}
