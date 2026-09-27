package plan

import (
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

func TestSingleCanonicalCampaignPlan(t *testing.T) {
	t.Parallel()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("campaign test path is unavailable")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	competing, err := CompetingPlans(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range competing {
		t.Errorf("competing live plan file %s; %s is the sole plan authority", path, filepath.ToSlash(Path))
	}

	// The .git-marker rule, pinned against a synthetic tree: a nested
	// checkout (marker file, as worktrees use, or marker directory) hides
	// its plan.json, while a plain nested directory's plan.json competes.
	synthetic := t.TempDir()
	write := func(relative, content string) {
		t.Helper()
		path := filepath.Join(synthetic, filepath.FromSlash(relative))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write(filepath.ToSlash(Path), "{}")
	write(".claude/worktrees/lane/.git", "gitdir: elsewhere")
	write(".claude/worktrees/lane/docs/plan.json", "{}")
	write("nested-checkout/.git/HEAD", "ref: refs/heads/main")
	write("nested-checkout/docs/plan.json", "{}")
	write("plain/docs/plan.json", "{}")
	competing, err = CompetingPlans(synthetic)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(competing, []string{"plain/docs/plan.json"}) {
		t.Fatalf("synthetic competing plans = %v, want only the plain nested copy", competing)
	}
}

// campaignStructuralViolations reports the campaign policy the doctrine prose
// used to pin -- capability freeze, the setverify machine check, and valid
// dependency and completion structure -- as the structure and behavior it
// protects rather than as phrases, so rewording the doctrine cannot pass while
// breaking a protected invariant cannot: the plan validates (dependency and
// completion structure), every open step carries a machine-checked verify (the
// setverify contract), and no step declares a capability (the validation
// campaign is capability-frozen -- it introduces no model family, workflow,
// optimizer, or orchestration framework). The UNAVAILABLE-not-credited outcome
// contract stays owned by the completion authority and its tests, not doctrine
// prose.
func campaignStructuralViolations(document Plan) []string {
	var violations []string
	if err := Validate(document); err != nil {
		violations = append(violations, "invalid plan structure: "+err.Error())
	}
	for _, item := range document.Items {
		if strings.HasPrefix(item.ID, "merge-") {
			continue
		}
		for _, step := range item.Steps {
			id := item.ID + "/" + step.ID
			if step.Status == StatusOpen && strings.TrimSpace(step.Verify) == "" {
				violations = append(violations, "step "+id+" has no machine-checked verify")
			}
			if len(step.Capabilities) != 0 {
				violations = append(violations, "step "+id+" declares capabilities "+strings.Join(step.Capabilities, ",")+"; the validation campaign is capability-frozen")
			}
		}
	}
	return violations
}

// TestStructuralCampaignPolicy holds that the campaign policy is enforced by
// structure, not doctrine prose: a campaign that keeps every historical doctrine
// phrase but breaks a protected invariant still fails, while rewording the
// doctrine without changing policy passes.
func TestStructuralCampaignPolicy(t *testing.T) {
	t.Parallel()
	// The doctrine phrases the tests used to pin, retained verbatim so a break
	// cannot hide behind them.
	const pinnedDoctrine = "capability freeze enforced by structure; a check records UNAVAILABLE rather than a pass; a step's machine check is set with plan -setverify"
	validCampaign := func() Plan {
		return Plan{
			Campaign: "Integrated validation", Doctrine: pinnedDoctrine,
			Items: []Item{{ID: "row", Title: "Row", Status: StatusOpen, Steps: []Step{
				{ID: "do", Title: "Do", Status: StatusOpen, Verify: "go test ./x -run '^TestX$'"},
			}}},
		}
	}
	if v := campaignStructuralViolations(validCampaign()); len(v) != 0 {
		t.Fatalf("valid campaign reported violations: %v", v)
	}
	reworded := validCampaign()
	reworded.Doctrine = "entirely different wording that keeps the same structural policy"
	if v := campaignStructuralViolations(reworded); len(v) != 0 {
		t.Fatalf("rewording the doctrine changed the policy verdict: %v", v)
	}
	for name, mutate := range map[string]func(*Plan){
		"forbidden capability expansion": func(p *Plan) { p.Items[0].Steps[0].Capabilities = []string{"new-orchestration-framework"} },
		"missing executable acceptance":  func(p *Plan) { p.Items[0].Steps[0].Verify = "" },
		"invalid dependency":             func(p *Plan) { p.Items[0].Steps[0].DependsOn = []string{"unproven"} },
	} {
		t.Run(name, func(t *testing.T) {
			broken := validCampaign()
			mutate(&broken)
			if v := campaignStructuralViolations(broken); len(v) == 0 {
				t.Fatalf("%s passed despite retaining every doctrine phrase", name)
			}
		})
	}
}

// The gate separately proves the exact parent and prunes this temporary row.
// Campaign closeout therefore cannot depend on it without changing the target plan.
func preparedMergeBoundary(item Item) bool {
	return preparedMergeShape(item) &&
		(item.Steps[0].Verify == "go run ./cmd/compatibility -check" ||
			strings.HasPrefix(item.Steps[0].Verify, "go run ./cmd/compatibility -check && "))
}

func preparedMergeShape(item Item) bool {
	revision, found := strings.CutPrefix(item.ID, "merge-")
	return found && len(revision) == 12 && strings.Trim(revision, "0123456789abcdef") == "" &&
		item.Status == StatusOpen && len(item.Steps) == 1 && item.Steps[0].ID == "do" &&
		item.Steps[0].Status == StatusOpen
}

func TestPreparedMergeBoundaryVerification(t *testing.T) {
	t.Parallel()
	item := Item{ID: "merge-fb7de0423660", Status: StatusOpen, Steps: []Step{{ID: "do", Status: StatusOpen}}}
	for _, test := range []struct {
		verify string
		want   bool
	}{
		{"go run ./cmd/compatibility -check", true},
		{"go run ./cmd/compatibility -check && go test ./internal/server -run '^TestConversationHistoryPaging$' -count=1", true},
		{"go test ./internal/server -run '^TestConversationHistoryPaging$' -count=1", false},
		{"go run ./cmd/compatibility -check || go test ./internal/server", false},
	} {
		item.Steps[0].Verify = test.verify
		if got := preparedMergeBoundary(item); got != test.want {
			t.Errorf("verification %q: prepared merge = %v, want %v", test.verify, got, test.want)
		}
	}
}

func loadCampaignPlan(t *testing.T) Plan {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("campaign test path is unavailable")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	document, err := Load(filepath.Join(root, filepath.FromSlash(Path)))
	if err != nil {
		t.Fatal(err)
	}
	if err := Validate(document); err != nil {
		t.Fatalf("validate campaign plan: %v", err)
	}
	return document
}

func retainedCampaignStep(document Plan, id string) (Step, bool) {
	itemID, stepID, ok := strings.Cut(id, "/")
	if !ok {
		return Step{}, false
	}
	for _, item := range document.Items {
		if item.ID == itemID {
			for _, step := range item.Steps {
				if step.ID == stepID {
					return step, true
				}
			}
		}
	}
	return Step{}, false
}

func campaignDependsOn(document Plan, source, target string, seen map[string]bool) bool {
	if seen[source] {
		return false
	}
	seen[source] = true
	step, found := retainedCampaignStep(document, source)
	if !found {
		return false
	}
	for _, dependency := range step.DependsOn {
		if dependency == target || campaignDependsOn(document, dependency, target, seen) {
			return true
		}
	}
	return false
}
