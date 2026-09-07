package testevidence

import (
	"context"
	"errors"
	"fmt"
	"io"

	"overgo/internal/processcontrol"
)

// RunGoTestCommand drains a test process through the bounded evidence parser.
// The caller owns command arguments, environment, diagnostic budget and verdict.
func RunGoTestCommand(ctx context.Context, command processcontrol.Command, short bool, diagnosticBytes int, observe func(string, bool) error) (GoTestReport, error) {
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	observe, flush := queuedPackageUpdates(observe)
	reader, writer := io.Pipe()
	defer reader.Close()
	command.Stdout, command.Stderr = writer, writer
	finished := make(chan error, 1)
	go func() {
		receipt, err := processcontrol.Run(ctx, command)
		if receipt.ExitCode != 0 {
			err = errors.Join(err, fmt.Errorf("test process exited with status %d", receipt.ExitCode))
		}
		writer.Close()
		finished <- err
	}()
	report, parseErr := GoTestJSONReader(reader, short, diagnosticBytes, observe)
	if parseErr != nil {
		cancel(parseErr)
	}
	// Even an oversized or unreadable event must not strand a writer on a full pipe.
	_, drainErr := io.Copy(io.Discard, reader)
	return report, errors.Join(parseErr, drainErr, <-finished, flush())
}
