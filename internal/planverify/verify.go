// Package planverify executes plan acceptance commands and validates their
// evidence with the immutable classifier contracts in runrecord.
package planverify

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/gitauthority"
	"overgo/internal/processcontrol"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
)

// ErrFailed marks a verifier whose command exited unsuccessfully, so a caller
// can tell a failed check from an absent one without reading the message.
var ErrFailed = errors.New("FAILED")

// ErrAssertion marks a failed verifier in which a test failed. A go test run
// that failed before any test ran, its package not building, and a command
// that runs no go test are ErrFailed alone.
var ErrAssertion = fmt.Errorf("%w on an assertion", ErrFailed)

// ErrVacuous marks a verifier whose command succeeded without running
// anything, which is what a check naming a test that does not exist does.
var ErrVacuous = errors.New("VACUOUS")

// Execute runs one plan verifier from directory and validates that a
// successful process produced non-vacuous evidence. It runs the command once.
// Overrides bind caller-owned inputs without changing the acceptance command.
func Execute(ctx context.Context, directory, command string, overrides []string) (runrecord.VerdictClass, error) {
	if ctx == nil {
		return "", errors.New("verification context is nil")
	}
	environment := append(gitauthority.RepositoryEnvironment(), overrides...)
	command = strings.TrimSpace(command)
	if command == "" {
		return "", errors.New("verification command is empty")
	}
	shell, err := clioptions.POSIXShell()
	if err != nil {
		return "", err
	}
	parsed, err := runrecord.ParseVerify(command)
	if err != nil {
		return "", err
	}
	if slices.ContainsFunc(parsed.Segments, handsAShellAString) {
		return "", fmt.Errorf("%w: the check hands a shell a command string, which hides what it runs from the parse and its evidence from the check; run each command as its own", ErrFailed)
	}
	structuredGoTest := len(parsed.GoTests()) != 0
	executable := parsed.RunCommand(command)
	first, err := executeShell(ctx, shell, directory, executable, environment)
	if err != nil {
		detail, failed := clioptions.Tail(first, clioptions.DiagnosticTailBytes), ErrFailed
		if failures := testevidence.FailureSummary(first); failures != "" {
			// The structured summary names the failed tests and their assertion
			// tails; it replaces the raw output wall, and the exit cause above
			// stays the authority. A failed test is named "package: test", a
			// package that failed before any test by its path alone.
			detail = testevidence.FailedTests + failures
			if named, _, _ := strings.Cut(failures, "\n"); strings.Contains(named, ": ") {
				failed = ErrAssertion
			}
		}
		return "", fmt.Errorf("%w: %v: %s", failed, err, detail)
	}
	if structuredGoTest {
		err = testevidence.VerifyGoTestEvidence(command, first)
	} else {
		// The parse found no go test in the command, whatever its text names.
		err = testevidence.VerifyOutput(false, first)
	}
	if err != nil {
		return "", fmt.Errorf("%w: %v", ErrVacuous, err)
	}
	// A check runs once (cheap-first/form): a second run of every host check, held to agree with the first, doubled
	// each check's time, and no refusal recorded nor any commit names a run that disagreed.
	return runrecord.ClassifyVerifyCommand(command), nil
}

// handsAShellAString says a segment runs a shell on a command string (sh -c, bash -c): what the string runs, a go
// test among it, is a word to the parse, so neither its -json evidence nor its tests' passing is read.
func handsAShellAString(segment runrecord.VerifySegment) bool {
	return len(segment.Fields) > 1 && slices.Contains([]string{"sh", "bash"}, segment.Fields[0]) && slices.Contains(segment.Fields[1:], "-c")
}

func executeShell(ctx context.Context, shell, directory, command string, environment []string) (string, error) {
	var output bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: shell, Args: []string{"-c", command}, Dir: directory, Env: environment,
		Stdout: &output, Stderr: &output,
	})
	if err != nil {
		return output.String(), err
	}
	if receipt.ExitCode != 0 {
		return output.String(), fmt.Errorf("exit code %d", receipt.ExitCode)
	}
	return output.String(), nil
}
