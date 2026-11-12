package processcontrol

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"overgo/internal/envsetting"
)

// TurnEnvironment carries the turns a process holds to the processes it starts, each turn's name to its holder's
// process id: a process started while its holder has the turn uses it in the holder's place, as
// quietmachine.HeldEnvironment does for the quiet machine, so a child its holder waits on never waits for its parent.
const TurnEnvironment = "OVERGO_TURNS_HELD"

// turnRecord is what a holder writes where a waiter reads it: its process, what it holds the turn for and since when.
type turnRecord struct {
	Process uint32    `json:"process"`
	Purpose string    `json:"purpose"`
	Since   time.Time `json:"since"`
}

// turnHold is this process's place on one turn: the holds taken on it, and how the place is given up once none is
// left (nothing, where the process rides its holder's place).
type turnHold struct {
	taken chan struct{}
	err   error
	holds int
	give  func() error
}

var (
	turnMu sync.Mutex
	turns  = map[string]*turnHold{}
)

// TakeTurn takes the turn named, one place every user of a resource queues for (a card's, taken where a CUDA context
// is created): the process holds it until the release it returns runs as many times as TakeTurn was called, or until
// the process ends, which the system learns as the turn's abandonment and hands the turn on. While another process
// holds it the call waits, saying once on output which process holds it, for what and since when, and when it is
// taken, how long it waited; nothing ends the wait on a clock. A process its holder started takes it in the holder's
// place without waiting (TurnEnvironment), and later holds in one process share its place.
func TakeTurn(name, purpose string, output io.Writer) (func() error, error) {
	if !validResourceName(name) {
		return nil, errors.New("processcontrol: invalid turn name")
	}
	turnMu.Lock()
	hold, placed := turns[name]
	if !placed {
		hold = &turnHold{taken: make(chan struct{})}
		turns[name] = hold
	}
	hold.holds++
	turnMu.Unlock()
	if !placed {
		hold.give, hold.err = takePlace(name, purpose, output)
		close(hold.taken)
	}
	<-hold.taken
	if hold.err != nil {
		turnMu.Lock()
		if hold.holds--; hold.holds == 0 && turns[name] == hold {
			delete(turns, name)
		}
		turnMu.Unlock()
		return nil, hold.err
	}
	return sync.OnceValue(func() error {
		turnMu.Lock()
		defer turnMu.Unlock()
		if hold.holds--; hold.holds > 0 {
			return nil
		}
		delete(turns, name)
		return hold.give()
	}), nil
}

// takePlace gives this process a place on the turn: its holder's, where the holder started it, else its own, waited
// for.
func takePlace(name, purpose string, output io.Writer) (func() error, error) {
	if holder, riding := heldTurns()[name]; riding {
		if record, held, err := turnHolder(name); err != nil {
			return nil, err
		} else if held && record.Process == holder {
			return func() error { return nil }, nil
		}
	}
	started := time.Now()
	waited := false
	if busy, err := ResourceTransactionHeld(name); err != nil {
		return nil, err
	} else if busy {
		waited = true
		record, held, err := turnHolder(name)
		if err != nil {
			return nil, err
		}
		holder := "a process that has written no record"
		if held {
			holder = fmt.Sprintf("%s for %s since %s", processLabel(record.Process), record.Purpose, record.Since.Format(time.DateTime))
		}
		fmt.Fprintf(output, "%s waits for the turn %s, held by %s\n", processLabel(uint32(os.Getpid())), name, holder)
	}
	leave, err := enterTurn(name)
	if err != nil {
		return nil, err
	}
	if waited {
		fmt.Fprintf(output, "%s took the turn %s after %s\n", processLabel(uint32(os.Getpid())), name, time.Since(started).Round(time.Millisecond))
	}
	unrecord, err := recordTurn(name, turnRecord{Process: uint32(os.Getpid()), Purpose: purpose, Since: time.Now()})
	if err != nil {
		return nil, errors.Join(err, leave())
	}
	if err := setHeldTurn(name, true); err != nil {
		return nil, errors.Join(err, unrecord(), leave())
	}
	return func() error { return errors.Join(setHeldTurn(name, false), unrecord(), leave()) }, nil
}

// enterTurn holds the turn's transaction (ResourceTransaction) on a thread of its own until leave runs: the system
// owns the hold, so the process's end hands it on.
func enterTurn(name string) (leave func() error, err error) {
	entered, leaving, left := make(chan struct{}), make(chan struct{}), make(chan error, 1)
	go func() {
		left <- ResourceTransaction(name, func() error {
			close(entered)
			<-leaving
			return nil
		})
	}()
	select {
	case <-entered:
		return sync.OnceValue(func() error { close(leaving); return <-left }), nil
	case err := <-left:
		return nil, err
	}
}

// heldTurns reads the turns this process's starter held for it (TurnEnvironment).
func heldTurns() map[string]uint32 {
	held := map[string]uint32{}
	if value := envsetting.Neutral(TurnEnvironment, "passes a turn's holder to the processes it starts"); value != "" {
		_ = json.Unmarshal([]byte(value), &held)
	}
	return held
}

// setHeldTurn names this process as the turn's holder to the processes it starts from now, or stops naming it.
func setHeldTurn(name string, held bool) error {
	turns := heldTurns()
	if held {
		turns[name] = uint32(os.Getpid())
	} else if turns[name] == uint32(os.Getpid()) {
		delete(turns, name)
	}
	if len(turns) == 0 {
		return os.Unsetenv(TurnEnvironment)
	}
	encoded, err := json.Marshal(turns)
	if err != nil {
		return err
	}
	return os.Setenv(TurnEnvironment, string(encoded))
}

// turnPurposeLimit bounds the purpose a holder records: a command line can run long, and the waiter prints it.
const turnPurposeLimit = 240

// CommandPurpose is what this process holds a turn for where it names nothing else: its command line, cut to a
// line's length.
func CommandPurpose() string {
	purpose := strings.Join(append([]string{filepath.Base(os.Args[0])}, os.Args[1:]...), " ")
	if len(purpose) > turnPurposeLimit {
		purpose = purpose[:turnPurposeLimit] + "..."
	}
	return purpose
}
