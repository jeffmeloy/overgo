package gate

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/repoanalysis"
	"overgo/internal/testutil"
)

func TestOwnerReuseRetainsDependentChecks(t *testing.T) {
	t.Parallel()
	for _, run := range []func() (bool, error){
		func() (bool, error) { return (&gateContext{}).stepTestOwners(t.Context()) },
		func() (bool, error) { return (&gateContext{}).stepTestDevice(t.Context()) },
		func() (bool, error) { return (&gateContext{}).stepTestRest(t.Context()) },
	} {
		if skipped, err := run(); skipped || err == nil {
			t.Fatal("unprepared execution was treated as inapplicable")
		}
	}
	if phaseReusesEvidence(testPlanCheckName) {
		t.Fatal("preparation was made reusable execution evidence")
	}
	root, _ := packageIdentityFixture(t)
	resources := t.TempDir()
	ready := filepath.Join(resources, "fixture-ready")
	ownerRuns := filepath.Join(resources, "owner-runs")
	files := map[string]string{
		".gitignore":          "tmp/\novergodb-store/\n",
		"NOTES.md":            "required runtime input",
		"app/app_test.go":     fmt.Sprintf("package app\nimport (\"os\";\"testing\")\nfunc TestOwner(t *testing.T) { f,err:=os.OpenFile(%q,os.O_CREATE|os.O_APPEND|os.O_WRONLY,0600);if err!=nil{t.Fatal(err)};defer f.Close();if _,err:=f.WriteString(\"owner\\n\");err!=nil{t.Fatal(err)} }\n", ownerRuns),
		"other/other_test.go": fmt.Sprintf("package other\nimport (\"os\";\"testing\")\nfunc TestDependent(t *testing.T) { if _,err:=os.ReadFile(\"../NOTES.md\");err!=nil{t.Fatal(err)};if _,err:=os.Stat(%q);err!=nil{t.Fatal(\"required fixture missing\")} }\n", ready),
		"third/third_test.go": "package third\nimport (\"os\";\"testing\")\nfunc TestSibling(t *testing.T) { if _,err:=os.ReadFile(\"../NOTES.md\");err!=nil{t.Fatal(err)} }\n",
	}
	for name, source := range files {
		testutil.WriteTextFile(t, root, name, source)
	}
	runGitFixture(t, root, "init", "-q")
	runGitFixture(t, root, "add", ".")
	environment := mustGateValue(discoverEnvironment(root))
	attempt := func() (*gateContext, map[string]automationcheck.DAGResult) {
		snapshot := mustGateValue(repoanalysis.DiscoverGo(root, "app", "dep", "other", "third"))
		g := &gateContext{repo: root, storePath: StorePath, environment: environment,
			paths: []string{"app/app_test.go", "NOTES.md"}, source: &snapshot}
		t.Cleanup(func() { _ = g.closeStore() })
		// Select the actual pipeline's package checks, preserving their real edges.
		planned := mustGateValue(automationcheck.Plan(g.pipelineChecks(), automationcheck.Impact{}))
		var selected []automationcheck.Invocation
		inputs := map[artifact.ID]artifact.ID{}
		for _, invocation := range planned {
			if !slices.Contains([]string{testPlanCheckName, testOwnersCheckName, testDeviceCheckName, testRestCheckName}, invocation.Check.Name) {
				continue
			}
			selected = append(selected, invocation)
			inputs[invocation.ID] = mustGateValue(g.phaseInputFingerprint(invocation.Check.Name))
		}
		if len(selected) != 4 || !slices.Contains(selected[1].Check.Dependencies, testPlanCheckName) {
			t.Fatal("pipeline owner reuse can bypass preparation")
		}
		cache := g.loadRetryCache()
		g.retryCache = &cache
		results, err := g.executeChecks(selected, map[string]bool{"acceptance": true}, inputs, &cache, nil)
		if err != nil {
			t.Fatal(err)
		}
		byName := map[string]automationcheck.DAGResult{}
		for _, result := range results {
			byName[result.Invocation.Check.Name] = result
		}
		if err := g.closeStore(); err != nil {
			t.Fatal(err)
		}
		return g, byName
	}
	first, results := attempt()
	if results[testOwnersCheckName].Err != nil || results[testOwnersCheckName].Evidence.Reused || results[testRestCheckName].Err == nil || !strings.Contains(results[testRestCheckName].Err.Error(), "required fixture missing") {
		t.Fatalf("first attempt did not isolate the dependent failure: %+v", results)
	}
	if first.testPlan == nil {
		t.Fatal("first attempt lost its package obligations")
	}
	restarted, results := attempt()
	if !results[testOwnersCheckName].Evidence.Reused || results[testRestCheckName].Err == nil || results[testRestCheckName].Evidence.Inapplicable {
		t.Fatalf("owner reuse suppressed the failed dependent after restart: %+v", results)
	}
	if restarted.testPlan == nil || restarted.testPlan.reused < 2 {
		t.Fatal("restart lost the independent passing packages")
	}
	assertOnlyDependent := func(g *gateContext, action string) {
		t.Helper()
		var executed []string
		for _, batch := range g.testExecutions {
			for _, execution := range batch.Executions {
				executed = append(executed, execution.Package)
				if execution.Action != action {
					t.Fatalf("dependent outcome=%s", execution.Action)
				}
			}
		}
		if !slices.Equal(executed, []string{"example/other"}) {
			t.Fatalf("repeated passed or omitted failed packages: %v", executed)
		}
	}
	assertOnlyDependent(restarted, "fail")
	// Resolve the external fixture without changing any source or acceptance.
	testutil.WriteTextFile(t, resources, filepath.Base(ready), "provided")
	repaired, results := attempt()
	for name, result := range results {
		if result.Err != nil {
			t.Fatalf("repaired %s: %v", name, result.Err)
		}
	}
	if !results[testOwnersCheckName].Evidence.Reused {
		t.Fatal("fixture repair repeated owner verification")
	}
	assertOnlyDependent(repaired, "pass")
	if data := mustGateValue(os.ReadFile(ownerRuns)); string(data) != "owner\n" {
		t.Fatalf("owner execution repeated: %q", data)
	}
}
