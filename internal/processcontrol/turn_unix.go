//go:build !windows

package processcontrol

import (
	"errors"
	"fmt"
)

func recordTurn(string, turnRecord) (func(turnRecord) error, func() error, error) {
	return nil, nil, errors.New("processcontrol: turns require Windows")
}

func turnHolder(string) (turnRecord, bool, error) {
	return turnRecord{}, false, errors.New("processcontrol: turns require Windows")
}

func turnUsers(string) ([]uint32, error) {
	return nil, errors.New("processcontrol: turns require Windows")
}

func markTurnWaiter(string) (func() error, error) {
	return nil, errors.New("processcontrol: turns require Windows")
}

func turnWaiters(string) ([]uint32, error) {
	return nil, errors.New("processcontrol: turns require Windows")
}

func processLabel(pid uint32) string { return fmt.Sprintf("pid %d", pid) }
