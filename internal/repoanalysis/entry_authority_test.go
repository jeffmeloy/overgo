package repoanalysis

import (
	"go/parser"
	"go/token"
	"path"
	"strings"
	"testing"

	"overgo/internal/gosyntax"
)

func TestEntryAuthorityScriptSuffixIsData(t *testing.T) {
	t.Parallel()
	rule, found := EntryAuthorityRuleFor(EntryAuthorityGoOnly)
	if !found {
		t.Fatal("missing Go-only rule")
	}
	for _, tc := range []struct {
		name, source string
		refuse       bool
	}{
		{"file classification", `import ("path/filepath"; "strings"); func classify(path string) bool { return strings.EqualFold(filepath.Ext(path), ".py") }`, false},
		{"source inventory", `func source(module string) string { return module + ".py" }`, false},
		{"named script", `const command = "execute.py"`, true},
		{"raw script name", "const command = `execute.py`", true},
		{"script invocation", `import "os/exec"; func run() { _ = exec.Command("python", "execute.py").Run() }`, true},
		{"extension invocation", `import runner "os/exec"; func run() { _ = runner.Command(".py").Run() }`, true},
		{"assembled script invocation", `import "os/exec"; func run(name string) { _ = exec.Command(name+".py").Run() }`, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			file, err := parser.ParseFile(token.NewFileSet(), "case.go", "package fixture\n"+tc.source, 0)
			if err != nil {
				t.Fatal(err)
			}
			if got := rule.Detect(file) > 0; got != tc.refuse {
				t.Fatalf("script classification=%v, want refusal=%v", got, tc.refuse)
			}
		})
	}
}

// Each case is independent: another violation cannot mask a broken detector.
func TestProductionAuthorityBoundaries(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name          string
		domain        EntryAuthorityDomain
		owner, source string
		exceptions    bool
	}{
		{"storage", EntryAuthorityStorage, "internal/overgodb", `import "os";func bypass()error{return os.WriteFile("overgodb.log",nil,0o644)}`, false},
		{"tool", EntryAuthorityTool, "internal/agenttool", `import "overgo/internal/agenttool";func bypass()agenttool.Manual{return agenttool.Manual{Name:"unregistered"}}`, false},
		{"promotion", EntryAuthorityPromotion, "internal/modelrecipe", `func bypass()string{return "recipe.active."+"model/task"}`, false},
		{"model", EntryAuthorityModel, "internal/modelrecipe", `import "overgo/internal/modelrecipe";func bypass()modelrecipe.ModelPrototype{return modelrecipe.ModelPrototype{Version:1}}`, false},
		{"routing", EntryAuthorityRouting, "internal/modelrecipe", `import "overgo/internal/modelrecipe";func bypass()modelrecipe.RecipeRoutingDecision{return modelrecipe.RecipeRoutingDecision{Derivation:"rogue"}}`, false},
		{"efficiency", EntryAuthorityEfficiency, "internal/runrecord", `import "overgo/internal/runrecord";func bypass()runrecord.EfficiencyTrace{return runrecord.EfficiencyTrace{Version:1}}`, false},
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
			snapshot, err := (gosyntax.Snapshot{}).Overlay(map[string][]byte{rogue: source})
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
				owned, err := (gosyntax.Snapshot{}).Overlay(map[string][]byte{path.Join(rule.OwnerPaths[0], "owned.go"): source})
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
