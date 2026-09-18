package gate

import (
	"context"
	"fmt"
	"strings"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/processcontrol"
)

// runLaneCommand runs one lane under the check's context with the source
// environment: the lane's whole process tree, its servers included, ends
// with the run, and the receipt reports how it ended.
func (g *gateContext) runLaneCommand(ctx context.Context, root, name string, args ...string) (string, error) {
	environment, err := g.sourceEnvironment()
	if err != nil {
		return "", err
	}
	receipt, err := superviseLane(ctx, environment, root, name, args...)
	if err != nil {
		return "", err
	}
	g.note(strings.Join(args, " ") + ": " + receipt)
	return receipt, nil
}

// superviseLane contains the lane's tree in a job that ends with the gate
// process, terminates it when the context ends, and waits for its terminal
// state before returning: a lane server cannot outlive its run on any exit
// path, so the tree's device claims end with the run. A child Windows
// could not start is launched again, as the plain command runner does.
func superviseLane(ctx context.Context, environment []string, root, name string, args ...string) (string, error) {
	output := clioptions.NewTailWriter(clioptions.DiagnosticTailBytes)
	var receipt processcontrol.Receipt
	var err error
	for range processStartAttempts {
		output.Reset()
		receipt, err = processcontrol.Run(ctx, processcontrol.Command{
			Path: name, Args: args, Dir: root, Env: environment, Stdout: output, Stderr: output,
		})
		if err != nil || receipt.ExitCode != windowsProcessStartFailure {
			break
		}
	}
	command := name + " " + strings.Join(args, " ")
	if cause := context.Cause(ctx); cause != nil {
		return "", fmt.Errorf("%s: lane tree terminated (exit=%d tree_terminated=%t): %w", command, receipt.ExitCode, receipt.TreeTerminated, cause)
	}
	if err != nil {
		return "", fmt.Errorf("%s: %w: %s", command, err, output.Tail())
	}
	if receipt.ExitCode != 0 {
		return "", fmt.Errorf("%s: exit status %d: %s", command, receipt.ExitCode, output.Tail())
	}
	return laneReceipt(receipt), nil
}

// laneReceipt renders the terminal receipt the lane's evidence records.
func laneReceipt(receipt processcontrol.Receipt) string {
	return fmt.Sprintf("lane tree ended: exit=%d tree_terminated=%t wall=%s; its device claims ended with it",
		receipt.ExitCode, receipt.TreeTerminated, time.Duration(receipt.WallNS).Round(time.Millisecond))
}
