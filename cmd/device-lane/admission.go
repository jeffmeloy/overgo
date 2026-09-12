package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strings"
	"time"

	"overgo/internal/cuda/driver"
	"overgo/internal/processcontrol"
)

// Match the gate's host batch admission budget; a lane cannot wait forever.
const deviceAdmissionBudget = 10 * time.Minute

// names the exhausted budget as the cancellation cause the report carries
var errAdmissionBudget = errors.New("device admission budget exhausted")

// names the host's device the way the gate's host batch admission does
func deviceIdentity() (string, error) {
	library, err := driver.Open()
	if err != nil {
		return "", err
	}
	defer library.Close()
	if err := library.Init(); err != nil {
		return "", err
	}
	info, err := library.DeviceInfo(0)
	if err != nil {
		return "", err
	}
	return info.UUID, nil
}

// builds a `go run` step into the lane's build directory so the step exits
// with its own status: go run reports a child's status as text and exits
// with one, which hides the typed contention status from the lane
func executableStep(ctx context.Context, root, buildDir string, step []string) ([]string, error) {
	program, found := strings.CutPrefix(strings.Join(step, " "), "go run ")
	if !found || strings.ContainsRune(program, ' ') {
		return step, nil
	}
	binary := filepath.Join(buildDir, filepath.Base(program))
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	var stdout, stderr bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: "go", Args: []string{"build", "-o", binary, program}, Dir: root, Env: os.Environ(),
		Stdout: &stdout, Stderr: &stderr,
	})
	if err != nil {
		return nil, err
	}
	if receipt.ExitCode != 0 {
		return nil, fmt.Errorf("build %s: exit code %d\n%s", program, receipt.ExitCode, stdout.String()+stderr.String())
	}
	return []string{binary}, nil
}

// Admit correctness work under the shared lease, then execute once. The
// admission budget ends at acquisition; execution retains the caller's context.
func runStepAdmitted(ctx context.Context, output io.Writer, name string, budget time.Duration, run func(context.Context) (int, error)) (int, error) {
	release, err := holdSharedLease(ctx, output, name, budget, func() (func() error, error) {
		return processcontrol.ShareResource(name)
	})
	if err != nil {
		return 0, err
	}
	code, err := run(ctx)
	return code, errors.Join(err, release())
}

// holds a shared lease around the correctness tests of a step, waiting for
// admission under the budget the way the gate's host batches do, so an
// exclusive holder is waited out instead of refusing every context the tests
// open, and a later exclusive claim waits for the step to release; the
// measurement tests never run beneath it, since their exclusive children
// would wait on their own ancestor
func holdSharedLease(ctx context.Context, output io.Writer, name string, budget time.Duration, share func() (func() error, error)) (func() error, error) {
	ctx, cancel := context.WithTimeoutCause(ctx, budget, errAdmissionBudget)
	defer cancel()
	began := time.Now()
	fmt.Fprintf(output, "[device] resource=%s mode=shared state=waiting budget=%s\n", name, budget)
	var release func() error
	err := processcontrol.AwaitResource(ctx, func() error {
		var err error
		release, err = share()
		return err
	})
	if err != nil {
		if errors.Is(err, processcontrol.ErrResourceBusy) {
			return nil, fmt.Errorf("device admission: %w", err)
		}
		return nil, err
	}
	fmt.Fprintf(output, "[device] resource=%s mode=shared state=admitted wait=%s\n", name, time.Since(began).Round(time.Millisecond))
	return func() error {
		err := release()
		if err == nil {
			fmt.Fprintf(output, "[device] resource=%s mode=shared state=not_busy scope=step\n", name)
		}
		return err
	}, nil
}

// runs a test step as its correctness run under the shared lease followed by
// its measurement run outside the lease; the first failing run ends the step
func runTestStep(ctx context.Context, output io.Writer, name string, step []string, run func(context.Context, []string) (int, error)) (int, error) {
	names, err := measurementTests(".", step)
	if err != nil {
		return 0, err
	}
	correctness, measurement := splitTestStep(step, names)
	code, err := runStepAdmitted(ctx, output, name, deviceAdmissionBudget, func(ctx context.Context) (int, error) {
		return run(ctx, correctness)
	})
	if err != nil || code != 0 || measurement == nil {
		return code, err
	}
	fmt.Fprintf(output, "[device] resource=%s mode=measurement tests=%d outside the shared lease\n", name, len(names))
	return run(ctx, measurement)
}

// The measurement helper every exclusive measurement test calls.
const measurementHelper = "MeasurementProcess"

// names the tests of the step's packages that run an exclusive measurement
// child, read from the test sources, so the lane can keep them outside its
// shared lease
func measurementTests(root string, step []string) ([]string, error) {
	var names []string
	for _, argument := range step {
		pattern, found := strings.CutPrefix(argument, "./")
		if !found {
			continue
		}
		directory := filepath.Join(root, filepath.FromSlash(pattern))
		recursive := false
		if suffix, ok := strings.CutSuffix(pattern, "/..."); ok {
			directory = filepath.Join(root, filepath.FromSlash(suffix))
			recursive = true
		}
		err := filepath.WalkDir(directory, func(path string, entry fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if entry.IsDir() {
				if path != directory && !recursive {
					return fs.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(entry.Name(), "_test.go") {
				return nil
			}
			file, err := parser.ParseFile(token.NewFileSet(), path, nil, parser.SkipObjectResolution)
			if err != nil {
				return err
			}
			for _, declaration := range file.Decls {
				function, ok := declaration.(*ast.FuncDecl)
				if !ok || function.Recv != nil || !strings.HasPrefix(function.Name.Name, "Test") || function.Body == nil {
					continue
				}
				if callsMeasurementHelper(function.Body) {
					names = append(names, function.Name.Name)
				}
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	slices.Sort(names)
	return slices.Compact(names), nil
}

func callsMeasurementHelper(body *ast.BlockStmt) bool {
	found := false
	ast.Inspect(body, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok || found {
			return !found
		}
		if selector, ok := call.Fun.(*ast.SelectorExpr); ok && selector.Sel.Name == measurementHelper {
			found = true
		}
		return !found
	})
	return found
}

// splits a test step into the correctness run, which skips the measurement
// tests and runs under the shared lease, and the measurement run, which
// selects only them and runs outside it; a step without measurement tests
// keeps its form and has no measurement run
func splitTestStep(step, measurement []string) (correctness, measurementRun []string) {
	selected := stepRunPattern(step)
	var chosen []string
	for _, name := range measurement {
		if selected == nil || selected.MatchString(name) {
			chosen = append(chosen, name)
		}
	}
	if len(chosen) == 0 {
		return step, nil
	}
	only := "^(" + strings.Join(chosen, "|") + ")$"
	correctness = append(slices.Clone(step), "-skip", only)
	measurementRun = append(slices.Clone(step), "-run", only)
	return correctness, measurementRun
}

// returns the step's own -run selection, or nil when it selects every test
func stepRunPattern(step []string) *regexp.Regexp {
	for index, argument := range step {
		if argument == "-run" && index+1 < len(step) {
			if pattern, err := regexp.Compile(step[index+1]); err == nil {
				return pattern
			}
		}
	}
	return nil
}
