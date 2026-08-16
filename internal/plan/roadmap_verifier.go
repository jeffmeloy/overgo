package plan

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
)

type roadmapVerifierExecution struct {
	Output             string
	CodeCommit         string
	Environment        artifact.ID
	EnvironmentContent artifact.Content
	MeasuredNS         uint64
}

func executeRoadmapVerifier(root, verifier string) (roadmapVerifierExecution, error) {
	shell, err := clioptions.POSIXShell()
	if err != nil {
		return roadmapVerifierExecution{}, err
	}
	started := time.Now()
	output, err := clioptions.CombinedOutputIn(root, os.Environ(), shell, "-c", testevidence.JSONCommand(verifier))
	measured := uint64(max(time.Since(started).Nanoseconds(), 1))
	if err != nil {
		return roadmapVerifierExecution{}, fmt.Errorf("roadmap verifier command failed: %w: %s", err, clioptions.Tail(output, 2000))
	}
	command := exec.Command("git", "rev-parse", "HEAD")
	command.Dir = root
	head, err := command.Output()
	if err != nil {
		return roadmapVerifierExecution{}, err
	}
	host, err := os.Hostname()
	if err != nil {
		host = "unknown"
	}
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: host, OS: runtime.GOOS, Arch: runtime.GOARCH, Device: "host", Backend: "go", Driver: runtime.Version(),
	})
	if err != nil {
		return roadmapVerifierExecution{}, err
	}
	content, err := environment.Content()
	if err != nil {
		return roadmapVerifierExecution{}, err
	}
	return roadmapVerifierExecution{
		Output: output, CodeCommit: strings.TrimSpace(string(head)), Environment: environment.ID,
		EnvironmentContent: content, MeasuredNS: measured,
	}, nil
}
