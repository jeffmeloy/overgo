package testutil

import (
	"errors"
	"fmt"

	"overgo/internal/operatoraction"
)

// Recoverable returns a blocked operation's failure only when the cause and recovery envelope are complete.
//   - a nil cause or an invalid block is refused as an ordinary error, so invalid recovery is never advertised to an operator
//   - the failure holds its own copy of block.
func Recoverable(cause error, block operatoraction.Block) error {
	if cause == nil {
		return errors.New("operator action: recoverable failure lacks a cause")
	}
	if err := block.Validate(); err != nil {
		return fmt.Errorf("operator action: recoverable failure: %w", err)
	}
	return &operatoraction.RecoverableError{Cause: cause, Block: block.Clone()}
}
