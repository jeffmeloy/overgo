package closurescan

import (
	"path"
	"strings"
	"testing"

	"overgo/internal/repoanalysis"
)

// Each case is independent: another violation cannot mask a broken detector.
func TestProductionAuthorityBoundaries(t *testing.T) {
	tests := []struct {
		name          string
		domain        EntryAuthorityDomain
		owner, source string
		exceptions    bool
	}{
		{"storage", EntryAuthorityStorage, "internal/overgodb", `import "os";func bypass()error{return os.WriteFile("overgodb.log",nil,0o644)}`, false},
		{"process", EntryAuthorityProcess, "internal/processcontrol", `import "os/exec";func bypass()error{return exec.Command("git","status").Run()}`, true},
		{"capability", EntryAuthorityCapability, "internal/inference", `import "overgo/internal/inference";func bypass()*inference.Runner{return &inference.Runner{}}`, false},
		{"tool", EntryAuthorityTool, "internal/agenttool", `import "overgo/internal/agenttool";func bypass()agenttool.Manual{return agenttool.Manual{Name:"unregistered"}}`, false},
		{"trigger", EntryAuthorityTrigger, "internal/workflowruntime", `import "overgo/internal/runrecord";func bypass()runrecord.WebhookDelivery{return runrecord.WebhookDelivery{}}`, false},
		{"promotion", EntryAuthorityPromotion, "internal/modelrecipe", `func bypass()string{return "recipe.active."+"model/task"}`, false},
		{"model", EntryAuthorityModel, "internal/modelrecipe", `import "overgo/internal/modelrecipe";func bypass()modelrecipe.ModelPrototype{return modelrecipe.ModelPrototype{Version:1}}`, false},
		{"routing", EntryAuthorityRouting, "internal/modelrecipe", `import "overgo/internal/modelrecipe";func bypass()modelrecipe.RecipeRoutingDecision{return modelrecipe.RecipeRoutingDecision{Derivation:"rogue"}}`, false},
		{"rollout", EntryAuthorityRollout, "internal/runrecord", `import "overgo/internal/runrecord";func bypass()runrecord.RolloutPlan{return runrecord.RolloutPlan{Version:1}}`, false},
		{"efficiency", EntryAuthorityEfficiency, "internal/runrecord", `import "overgo/internal/runrecord";func bypass()runrecord.EfficiencyTrace{return runrecord.EfficiencyTrace{Version:1}}`, false},
		{"safety", EntryAuthoritySafety, "internal/evaluation", `import "overgo/internal/evaluation";func bypass()evaluation.LiveSafetyWindow{return evaluation.LiveSafetyWindow{Version:1}}`, false},
		{"dll", EntryAuthorityGoOnly, "compiled Go registrations", `import "syscall";func bypass()*syscall.LazyDLL{return syscall.NewLazyDLL("rogue.dll")}`, true},
		{"interpreter", EntryAuthorityGoOnly, "compiled Go registrations", `import _ "github.com/dop251/goja"`, true},
		{"plugin", EntryAuthorityGoOnly, "compiled Go registrations", `import _ "plugin"`, true},
	}
	covered := map[EntryAuthorityDomain]bool{}
	for _, test := range tests {
		covered[test.domain] = true
		t.Run(test.name, func(t *testing.T) {
			rule, found := EntryAuthorityRuleFor(test.domain)
			if !found || !strings.Contains(rule.Owner, test.owner) || test.exceptions && len(rule.Exceptions) == 0 {
				t.Fatalf("rule authority: found=%t owner=%q exceptions=%d", found, rule.Owner, len(rule.Exceptions))
			}
			const rogue = "internal/rogue/rogue.go"
			source := []byte("package rogue\n" + test.source + "\n")
			snapshot, err := (repoanalysis.SourceSnapshot{}).Overlay(map[string][]byte{rogue: source})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := ValidateEntryAuthorities(snapshot, []EntryAuthorityRule{rule}); err == nil || !strings.Contains(err.Error(), rogue+" bypasses the owner") {
				t.Fatalf("bypass escaped its domain: %v", err)
			}
			// Synthetic exceptions exercise admission and stale-site rejection.
			rule.Exceptions = map[string]string{rogue: "test-owned direct site"}
			if _, err := ValidateEntryAuthorities(snapshot, []EntryAuthorityRule{rule}); err != nil {
				t.Fatalf("exact exception refused: %v", err)
			}
			for _, replacement := range [][]byte{nil, []byte("package rogue\n")} {
				clean, err := snapshot.Overlay(map[string][]byte{rogue: replacement})
				if err != nil {
					t.Fatal(err)
				}
				if _, err := ValidateEntryAuthorities(clean, []EntryAuthorityRule{rule}); err == nil || !strings.Contains(err.Error(), "no longer holds a direct site") {
					t.Fatalf("stale exception admitted: %v", err)
				}
			}
			// The owner keeps its right to construct its own values.
			rule.Exceptions = nil
			if len(rule.OwnerPaths) != 0 {
				owned, err := (repoanalysis.SourceSnapshot{}).Overlay(map[string][]byte{path.Join(rule.OwnerPaths[0], "owned.go"): source})
				if err != nil {
					t.Fatal(err)
				}
				if _, err := ValidateEntryAuthorities(owned, []EntryAuthorityRule{rule}); err != nil {
					t.Fatalf("owner refused: %v", err)
				}
			}
		})
	}
	for _, rule := range EntryAuthorityRules() {
		if !covered[rule.Domain] {
			t.Errorf("domain %s has no independent bypass case", rule.Domain)
		}
	}
}
