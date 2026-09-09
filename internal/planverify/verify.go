// Package planverify executes plan acceptance commands and validates their
// evidence with the immutable classifier contracts in testevidence.
package planverify

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/gitauthority"
	"overgo/internal/processcontrol"
	"overgo/internal/testevidence"
)

// Execute runs one plan verifier from directory and validates that a
// successful process produced non-vacuous evidence. Deterministic Go-test
// claims are repeated against the same working tree and must agree exactly.
// Overrides bind caller-owned inputs without changing the acceptance command.
func Execute(ctx context.Context, directory, command string, overrides []string) (testevidence.VerdictClass, error) {
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
	structuredGoTest := strings.Contains(command, "go test")
	executable := command
	if structuredGoTest {
		executable = testevidence.JSONCommand(command)
	}
	first, err := executeShell(ctx, shell, directory, executable, environment)
	if err != nil {
		detail := clioptions.Tail(first, clioptions.DiagnosticTailBytes)
		if failures := testevidence.FailureSummary(first); failures != "" {
			detail = "failed tests/packages: " + failures + "\n" + detail
		}
		return "", fmt.Errorf("FAILED: %w: %s", err, detail)
	}
	if structuredGoTest {
		err = testevidence.VerifyGoTestEvidence(command, first)
	} else {
		err = testevidence.VerifyOutput(command, first)
	}
	if err != nil {
		return "", fmt.Errorf("VACUOUS: %w", err)
	}
	verdict := testevidence.ClassifyVerifyCommand(command)
	if structuredGoTest && verdict == testevidence.VerdictBitwiseDeterministic {
		repeat, repeatErr := executeShell(ctx, shell, directory, executable, environment)
		if repeatErr != nil {
			return "", fmt.Errorf("REPEAT failed: %w: %s", repeatErr, clioptions.Tail(repeat, clioptions.DiagnosticTailBytes))
		}
		if err := testevidence.RepeatAgreementForCommand(command, first, repeat); err != nil {
			return "", fmt.Errorf("NOT deterministic: %w", err)
		}
	}
	return verdict, nil
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
