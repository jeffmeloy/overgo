package repoanalysis

import (
	"fmt"
	"go/ast"
	"go/token"
	"maps"
	"slices"
	"strings"
)

// FixedTimerBaselineFile is the reviewed census of wall-clock bounds the
// source still fixes at writing time. The ratchet lets it shrink only.
const FixedTimerBaselineFile = "docs/fixed_timer_baseline.json"

// FixedTimer counts, in one function, the waits bound to a duration the
// source fixes rather than one the caller declares.
type FixedTimer struct {
	File     string `json:"file"`
	Function string `json:"function"`
	Count    int    `json:"count"`
}

// FixedTimerBaseline is the reviewed census document.
type FixedTimerBaseline struct {
	Version int          `json:"version"`
	Doc     string       `json:"doc"`
	Timers  []FixedTimer `json:"timers"`
}

// timerCalls are the calls that bind a wait or a delay to a duration.
var timerCalls = map[string]map[string]bool{
	"context": {"WithTimeout": true, "WithTimeoutCause": true, "WithDeadline": true, "WithDeadlineCause": true},
	"time":    {"After": true, "Sleep": true, "NewTimer": true, "AfterFunc": true, "Tick": true, "NewTicker": true},
}

// FixedTimerCensus lists every timer call whose duration the source fixes:
// a literal, a time unit, a constant, or a variable initialised from those.
// A duration that arrives through a parameter, a field, a flag or another
// computed value is the caller's declaration and is not counted.
func FixedTimerCensus(snapshot SourceSnapshot) ([]FixedTimer, error) {
	counts := map[[2]string]int{}
	for _, file := range snapshot.Files {
		parsed, err := file.Syntax()
		if err != nil {
			return nil, fmt.Errorf("fixed timers: parse %s: %w", file.Path, err)
		}
		for _, declaration := range parsed.Decls {
			function, _ := declaration.(*ast.FuncDecl)
			name := authorityFunctionName(function)
			ast.Inspect(declaration, func(node ast.Node) bool {
				call, ok := node.(*ast.CallExpr)
				if !ok {
					return true
				}
				if duration, timer := timerDuration(call); timer && fixedDuration(duration, 0) {
					counts[[2]string{file.Path, name}]++
				}
				return true
			})
		}
	}
	timers := make([]FixedTimer, 0, len(counts))
	for key, count := range counts {
		timers = append(timers, FixedTimer{File: key[0], Function: key[1], Count: count})
	}
	slices.SortFunc(timers, func(left, right FixedTimer) int {
		return strings.Compare(left.File+"\x00"+left.Function, right.File+"\x00"+right.Function)
	})
	return timers, nil
}

// AdmitFixedTimers refuses a census that adds a timer the baseline does not
// list or exceeds a listed count, and a baseline that still lists a timer
// the source no longer holds: the baseline follows every removal down.
func AdmitFixedTimers(baseline FixedTimerBaseline, census []FixedTimer) error {
	allowed := make(map[[2]string]int, len(baseline.Timers))
	for _, timer := range baseline.Timers {
		allowed[[2]string{timer.File, timer.Function}] = timer.Count
	}
	for _, timer := range census {
		key := [2]string{timer.File, timer.Function}
		limit, known := allowed[key]
		if !known || timer.Count > limit {
			return fmt.Errorf("fixed timer exceeds the reviewed census: %s %s %d -> %d; wait on the operation's own progress under caller cancellation", timer.File, timer.Function, limit, timer.Count)
		}
		if timer.Count < limit {
			return fmt.Errorf("fixed timer census fell below the baseline: %s %s %d -> %d; lower %s", timer.File, timer.Function, limit, timer.Count, FixedTimerBaselineFile)
		}
		delete(allowed, key)
	}
	for key := range maps.Keys(allowed) {
		return fmt.Errorf("fixed timer baseline lists a timer the source no longer holds: %s %s; lower %s", key[0], key[1], FixedTimerBaselineFile)
	}
	return nil
}

// timerDuration returns the duration argument of a timer call.
func timerDuration(call *ast.CallExpr) (ast.Expr, bool) {
	selector, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return nil, false
	}
	pkg, ok := selector.X.(*ast.Ident)
	if !ok || !timerCalls[pkg.Name][selector.Sel.Name] {
		return nil, false
	}
	switch {
	case pkg.Name == "context" && len(call.Args) >= 2:
		return call.Args[1], true
	case pkg.Name == "time" && len(call.Args) >= 1:
		return call.Args[0], true
	}
	return nil, false
}

const fixedDurationDepth = 8

// fixedDuration reports an expression built only from literals, time units,
// constants and variables initialised from those.
func fixedDuration(expression ast.Expr, depth int) bool {
	if depth > fixedDurationDepth {
		return false
	}
	switch typed := expression.(type) {
	case *ast.BasicLit:
		return typed.Kind == token.INT || typed.Kind == token.FLOAT
	case *ast.ParenExpr:
		return fixedDuration(typed.X, depth+1)
	case *ast.UnaryExpr:
		return fixedDuration(typed.X, depth+1)
	case *ast.BinaryExpr:
		return fixedDuration(typed.X, depth+1) && fixedDuration(typed.Y, depth+1)
	case *ast.CallExpr:
		// A conversion such as time.Duration(n) keeps the operand's nature.
		return len(typed.Args) == 1 && fixedDuration(typed.Args[0], depth+1)
	case *ast.SelectorExpr:
		pkg, ok := typed.X.(*ast.Ident)
		return ok && pkg.Name == "time" && pkg.Obj == nil
	case *ast.Ident:
		if typed.Obj == nil {
			return false
		}
		switch typed.Obj.Kind {
		case ast.Con:
			return true
		case ast.Var:
			spec, ok := typed.Obj.Decl.(*ast.ValueSpec)
			if !ok {
				return false
			}
			index := slices.IndexFunc(spec.Names, func(name *ast.Ident) bool { return name.Name == typed.Name })
			return index >= 0 && index < len(spec.Values) && fixedDuration(spec.Values[index], depth+1)
		}
	}
	return false
}
