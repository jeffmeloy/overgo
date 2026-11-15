package repoanalysis

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
)

// clockOwners are the production files whose clock may end work, each with the limit it applies: one a person set,
// none by default, or a network peer's. Inside the machine a clock measures, reports and schedules; it never ends or
// refuses work (the owner's rule of 2026-11-12).
var clockOwners = map[string]string{
	"internal/clioptions/limit.go":     "Limited: the limit a flag of the user's sets, none where zero (limitSetters)",
	"cmd/hf-hub/main.go":               "a network peer, the hub, under the user's -timeout, none by default",
	"internal/server/hub_workspace.go": "a network peer, the hub, under the timeout the user's request names, none by default",
	"internal/server/server.go":        "a request under the operator's -request-timeout, none by default",
	"cmd/server/main.go":               "the drain under the operator's -request-timeout, none by default",
	"internal/modelswap/supervisor.go": "an unreferenced child stopped after the user's cmd/swap -idle, none by default",
}

// limitSetters are the production files that hand clioptions.Limited a limit, each with the flag of the user's it
// carries.
var limitSetters = map[string]string{
	"cmd/evaluate/main.go":          "-budget, on a speech or transcription evaluation",
	"cmd/evaluate/all.go":           "-budget, on an -all pass, sliced across its models, and on a worker",
	"cmd/longform/main.go":          "-budget and -model-budget",
	"cmd/validate/smoke.go":         "-smoke -budget",
	"internal/webuilane/browser.go": "webui-lane -wait, through WaitBudgetEnvironment",
}

// contextClocks are the context package's functions that end a context by a clock.
var contextClocks = []string{"WithTimeout", "WithTimeoutCause", "WithDeadline", "WithDeadlineCause"}

// TestNoClockEndsWorkInsideTheMachine holds skill.md's rule that inside the machine a clock measures, reports and
// schedules and never ends or refuses work: a context ended by a clock, or a time.AfterFunc whose function cancels,
// stops, kills or closes, stands only in clockOwners; a limit handed to clioptions.Limited comes only from
// limitSetters' flags; no go test a command runs names a timeout but -timeout=0; and no open plan check names one, a
// check that names none running with none (runrecord.VerifyCommand.RunCommand). Its fixture pins the reading of a
// clock: a timeout, a deadline and an AfterFunc that stops count, and an AfterFunc that reports does not.
func TestNoClockEndsWorkInsideTheMachine(t *testing.T) {
	t.Parallel()
	const fixture = `package p

func waits(ctx context.Context, cancel func()) {
	context.WithTimeoutCause(ctx, time.Hour, nil)
	context.WithDeadline(ctx, time.Now())
	time.AfterFunc(time.Hour, func() { cancel() })
	time.AfterFunc(time.Hour, report)
	clioptions.Limited(ctx, budget, nil)
	_ = []string{"-timeout=10m", "-timeout=0"}
}
`
	parsed := testcheck.Must(parser.ParseFile(token.NewFileSet(), "p.go", fixture, parser.SkipObjectResolution))(t)
	if clocks, limits, timeouts := clockReadings(parsed); clocks != 3 || limits != 1 || !slices.Equal(timeouts, []string{"-timeout=10m"}) {
		t.Fatalf("the fixture reads %d clocks, %d limits and timeouts %q; want the timeout's, the deadline's and the stopping AfterFunc's, one limit and -timeout=10m", clocks, limits, timeouts)
	}
	clocked, limited := map[string]bool{}, map[string]bool{}
	for _, tree := range []string{"cmd", "internal"} {
		testcheck.NoError(t, filepath.WalkDir(filepath.Join(checkoutRoot, tree), func(file string, entry fs.DirEntry, err error) error {
			if err != nil || entry.IsDir() || !strings.HasSuffix(file, ".go") || strings.HasSuffix(file, "_test.go") || strings.Contains(filepath.ToSlash(file), "/testdata/") {
				return err
			}
			relative := filepath.ToSlash(testcheck.Must(filepath.Rel(checkoutRoot, file))(t))
			source := testcheck.Must(os.ReadFile(file))(t)
			syntax, err := parser.ParseFile(token.NewFileSet(), file, source, parser.SkipObjectResolution)
			if err != nil {
				return err
			}
			clocks, limits, timeouts := clockReadings(syntax)
			clocked[relative], limited[relative] = clocks > 0, limits > 0
			for _, timeout := range timeouts {
				t.Errorf("%s runs a go test with %s, which ends it by a clock; run it with -timeout=0", relative, timeout)
			}
			return nil
		}))
	}
	held := func(kind string, found map[string]bool, declared map[string]string) {
		for _, file := range slices.Sorted(maps.Keys(found)) {
			if _, listed := declared[file]; found[file] && !listed {
				t.Errorf("%s holds a %s outside the limits a person set or a network peer's; wait on the event, or take the user's flag (clioptions.Limited)", file, kind)
			}
		}
		for file := range declared {
			if !found[file] {
				t.Errorf("%s is declared to hold a %s it no longer holds; remove it from the list", file, kind)
			}
		}
	}
	held("clock that ends work", clocked, clockOwners)
	held("limit for clioptions.Limited", limited, limitSetters)
	document := testcheck.Must(plan.Load(filepath.Join(checkoutRoot, filepath.FromSlash(plan.Path))))(t)
	for _, item := range document.Items {
		for _, step := range item.Steps {
			parsed, err := runrecord.ParseVerify(step.Verify)
			if err != nil {
				continue
			}
			for _, test := range parsed.GoTests() {
				if slices.ContainsFunc(test.Flags, func(flag string) bool { return strings.HasPrefix(flag, "-timeout=") && flag != "-timeout=0" }) {
					t.Errorf("%s/%s's check runs a go test under a timeout, which ends it by a clock; set it to -timeout=0 or name none", item.ID, step.ID)
				}
			}
		}
	}
}

// clockReadings counts a file's clocks that end work, its limits handed to clioptions.Limited, and the go test
// timeouts it names other than -timeout=0.
func clockReadings(file *ast.File) (clocks, limits int, timeouts []string) {
	ast.Inspect(file, func(node ast.Node) bool {
		switch typed := node.(type) {
		case *ast.CallExpr:
			selector, isSelector := typed.Fun.(*ast.SelectorExpr)
			switch {
			case !isSelector:
			case isNamed(selector.X, "context") && slices.Contains(contextClocks, selector.Sel.Name):
				clocks++
			case isNamed(selector.X, "time") && selector.Sel.Name == "AfterFunc" && len(typed.Args) == 2 && endsWork(typed.Args[1]):
				clocks++
			case isNamed(selector.X, "clioptions") && selector.Sel.Name == "Limited":
				limits++
			}
		case *ast.BasicLit:
			text, _ := strconv.Unquote(typed.Value)
			if typed.Kind == token.STRING && strings.HasPrefix(text, "-timeout=") && text != "-timeout=0" {
				timeouts = append(timeouts, text)
			}
		}
		return true
	})
	return clocks, limits, timeouts
}

// endsWork says a function a timer runs cancels, stops, kills or closes: a name in it says so.
func endsWork(function ast.Expr) bool {
	ends := false
	ast.Inspect(function, func(node ast.Node) bool {
		if ident, isIdent := node.(*ast.Ident); isIdent {
			name := strings.ToLower(ident.Name)
			ends = ends || strings.Contains(name, "cancel") || strings.Contains(name, "stop") || strings.Contains(name, "kill") || strings.Contains(name, "close")
		}
		return !ends
	})
	return ends
}
