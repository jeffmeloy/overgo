package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"overgo/internal/processcontrol"
)

// builds a `go run` step into the lane's build directory so the step exits
// with its own status: go run reports a child's status as text and exits
// with one of its own. Each step's contexts wait for the card's turn as they
// are created and say for whom (card-turn), so the lane admits nothing itself.
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
