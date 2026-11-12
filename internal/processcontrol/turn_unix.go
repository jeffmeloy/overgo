//go:build !windows

package processcontrol

import (
	"errors"
	"fmt"
)

func recordTurn(string, turnRecord) (func() error, error) {
	return nil, errors.New("processcontrol: turns require Windows")
}

func turnHolder(string) (turnRecord, bool, error) {
	return turnRecord{}, false, errors.New("processcontrol: turns require Windows")
}

func processLabel(pid uint32) string { return fmt.Sprintf("pid %d", pid) }
