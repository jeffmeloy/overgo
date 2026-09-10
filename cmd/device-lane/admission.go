package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"overgo/internal/processcontrol"
)

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

// Match the gate's host batch admission budget; a lane cannot wait forever.
const deviceAdmissionBudget = 10 * time.Minute

// names the exhausted budget as the cancellation cause the report carries
var errAdmissionBudget = errors.New("device admission budget exhausted")

// runs one step again while it exits with the typed contention status, so a
// foreign holder is waited out under the lane budget and each exclusive
// hold stays as short as the step itself, leaving the browser lane's shared
// use of the device undisturbed between steps
func runStepAdmitted(ctx context.Context, output io.Writer, name string, run func() (int, error)) (int, error) {
	ctx, cancel := context.WithTimeoutCause(ctx, deviceAdmissionBudget, errAdmissionBudget)
	defer cancel()
	began := time.Now()
	waiting := false
	code := 0
	err := processcontrol.AwaitResource(ctx, func() error {
		var err error
		code, err = run()
		if err != nil || code != processcontrol.ResourceBusyExitCode {
			return err
		}
		if !waiting {
			waiting = true
			fmt.Fprintf(output, "[device] resource=%s mode=exclusive state=waiting budget=%s\n", name, deviceAdmissionBudget)
		}
		return processcontrol.ErrResourceBusy
	})
	if err != nil {
		if errors.Is(err, processcontrol.ErrResourceBusy) {
			return code, fmt.Errorf("device admission: %w", err)
		}
		return code, err
	}
	if waiting {
		fmt.Fprintf(output, "[device] resource=%s mode=exclusive state=admitted wait=%s\n", name, time.Since(began).Round(time.Millisecond))
	}
	return code, nil
}
