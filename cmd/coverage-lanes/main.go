// Command coverage-lanes reports verification depth by lane, so an
// aggregate number cannot hide a weak execution path: the hermetic lane
// runs the short suites with statement coverage and holds each package to
// its reviewed floor, while the model, CUDA, and browser lanes report
// whether their prerequisites are present on this machine -- their depth is
// measured by their own gated suites, never blended into the hermetic
// number.
package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"os"
	"regexp"
	"strconv"

	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/processcontrol"
	"overgo/internal/testevidence"
)

// coverageBaselineFile is the reviewed per-package hermetic coverage floor.
const coverageBaselineFile = "docs/coverage_lanes_baseline.json"

type coverageBaseline struct {
	Version  int                `json:"version"`
	Doc      string             `json:"doc"`
	Hermetic map[string]float64 `json:"hermetic"`
}

// gatedLanes names each non-hermetic lane beside the environment fact that
// unlocks it on a machine.
var gatedLanes = []struct{ lane, environment string }{
	{"installed-models", "OVERGO_QWEN3_MODEL"},
	{"cuda", "OVERGO_QWEN35_MODEL"},
	{"browser", "OVERGO_BROWSER"},
}

var coveragePattern = regexp.MustCompile(`coverage: ([0-9.]+)% of statements`)

func main() {
	clioptions.MainNamed("coverage-lanes", run)
}

func run() error {
	update := flag.Bool("update", false, "raise the hermetic floors to the measured coverage")
	repeat := flag.String("repeat", "", "run this package's short suite twice and refuse on repeat disagreement (flaky-command detection)")
	flag.Parse()
	if *repeat != "" {
		return repeatAgreement(*repeat)
	}
	baseline, err := loadCoverageBaseline()
	if err != nil {
		return err
	}
	failed := 0
	measured := map[string]float64{}
	for pkg, floor := range baseline.Hermetic {
		var output bytes.Buffer
		receipt, err := processcontrol.Run(context.Background(), processcontrol.Command{
			Path: "go", Args: []string{"test", pkg, "-short", "-count=1", "-cover"},
			Stdout: &output, Stderr: &output,
		})
		if err != nil {
			return err
		}
		if receipt.ExitCode != 0 {
			return fmt.Errorf("hermetic lane %s failed: %s", pkg, output.String())
		}
		match := coveragePattern.FindStringSubmatch(output.String())
		if match == nil {
			return fmt.Errorf("hermetic lane %s reported no coverage: %s", pkg, output.String())
		}
		coverage, err := strconv.ParseFloat(match[1], 64)
		if err != nil {
			return err
		}
		measured[pkg] = coverage
		verdict := "ok"
		if coverage < floor {
			verdict = "BELOW FLOOR"
			failed++
		}
		fmt.Printf("coverage-lanes: hermetic %s = %.1f%% (floor %.1f%%) %s\n", pkg, coverage, floor, verdict)
	}
	for _, gated := range gatedLanes {
		state := "unavailable (set " + gated.environment + ")"
		if os.Getenv(gated.environment) != "" {
			state = "available; run its gated suite for this lane's depth"
		}
		fmt.Printf("coverage-lanes: %s lane %s\n", gated.lane, state)
	}
	if *update {
		for pkg, coverage := range measured {
			if coverage > baseline.Hermetic[pkg] {
				baseline.Hermetic[pkg] = coverage
			}
		}
		if err := jsonfile.Write(coverageBaselineFile, baseline, clioptions.OutputFileMode); err != nil {
			return err
		}
		fmt.Println("coverage-lanes: floors raised to measured coverage")
	}
	if failed > 0 {
		return fmt.Errorf("%d hermetic package(s) below their coverage floor", failed)
	}
	return nil
}

// repeatAgreement runs one package's short suite twice and derives the
// repeat-agreement verdict: a test that changes outcome between identical
// runs is flaky, and flakiness is a finding here, never noise absorbed by
// a retry.
func repeatAgreement(pkg string) error {
	command := "go test " + pkg + " -short -json -count=1"
	first, err := goTestJSON(pkg)
	if err != nil {
		return err
	}
	second, err := goTestJSON(pkg)
	if err != nil {
		return err
	}
	if err := testevidence.RepeatAgreementForCommand(command, first, second); err != nil {
		return fmt.Errorf("repeat disagreement for %s: %w", pkg, err)
	}
	fmt.Printf("coverage-lanes: repeat agreement holds for %s\n", pkg)
	return nil
}

func goTestJSON(pkg string) (string, error) {
	var output bytes.Buffer
	receipt, err := processcontrol.Run(context.Background(), processcontrol.Command{
		Path: "go", Args: []string{"test", pkg, "-short", "-json", "-count=1"},
		Stdout: &output, Stderr: &output,
	})
	if err != nil {
		return "", err
	}
	if receipt.ExitCode != 0 {
		return "", fmt.Errorf("suite %s failed: %s", pkg, output.String())
	}
	return output.String(), nil
}

func loadCoverageBaseline() (coverageBaseline, error) {
	var baseline coverageBaseline
	if err := jsonfile.DecodeStrict(coverageBaselineFile, &baseline); err != nil {
		return coverageBaseline{}, err
	}
	if len(baseline.Hermetic) == 0 {
		return coverageBaseline{}, fmt.Errorf("coverage baseline declares no hermetic packages")
	}
	return baseline, nil
}
