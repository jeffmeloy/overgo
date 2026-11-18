package processcontrol

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"overgo/internal/envsetting"
)

// TurnEnvironment carries the turns a process holds to the processes it starts, each turn's name to its holder's
// process id: a process started while its holder has the turn uses it in the holder's place, as
// quietmachine.HeldEnvironment does for the quiet machine, so a child its holder waits on never waits for its parent.
const TurnEnvironment = "OVERGO_TURNS_HELD"

// CardTurnPrefix begins the name of every graphics card's turn (driver.CardTurn), by which a reader finds the card's
// holder without opening the card (TurnsHeld).
const CardTurnPrefix = "overgo-card-turn:"

// turnRecord is what a holder writes where a waiter reads it: the turn's name, its process, what it holds the turn for
// and since when, and, where it counts its work (TurnCount), how far it has come and when it got there.
type turnRecord struct {
	Name    string    `json:"name,omitzero"`
	Process uint32    `json:"process"`
	Purpose string    `json:"purpose"`
	Since   time.Time `json:"since"`
	Done    int       `json:"done,omitzero"`
	Total   int       `json:"total,omitzero"`
	Reached time.Time `json:"reached,omitzero"`
}

// turnHold is this process's place on one turn: the holds taken on it, how the place is given up once none is left,
// and how its count is recorded (nothing of either, where the process rides its holder's place).
type turnHold struct {
	taken chan struct{}
	err   error
	holds int
	give  func() error
	count func(done, total int) error
}

var (
	turnMu sync.Mutex
	turns  = map[string]*turnHold{}
)

// TakeTurn takes the turn named, one place every user of a resource queues for (a card's, taken where a CUDA context
// is created): the process holds it until the release it returns runs as many times as TakeTurn was called, or until
// the process ends, which the system learns as the turn's abandonment and hands the turn on. While another process
// holds it the call waits, saying on output which process holds it, for what and since when, again each time the turn
// passes to another, and when it is taken, how long it waited; nothing ends the wait on a clock. A process its holder started takes it in the holder's
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
		hold.give, hold.count, hold.err = takePlace(name, purpose, output)
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
		if hold.give == nil {
			return nil
		}
		return hold.give()
	}), nil
}

// TurnWaited says whether another process waits for a turn this process holds in its own place (TakeTurn): a holder
// that can give its turn up between pieces of work, as a reading between pictures, gives it up only to a waiter.
func TurnWaited() (bool, error) {
	turnMu.Lock()
	held := slices.Collect(maps.Keys(ownTurns()))
	turnMu.Unlock()
	for _, name := range held {
		if waiters, err := turnWaiters(name); err != nil || len(waiters) > 0 {
			return err == nil, err
		}
	}
	return false, nil
}

// YieldTurn gives the turn named, which this process holds in its own place, up to the processes waiting for it
// (release, as TakeTurn returned it), and returns once another process holds it, or once every process that waited as
// it was given up has ended: a holder that gives its turn up between pieces of work and takes it again for the next
// took it back first, as a waiter that has marked itself waiting has yet to wait on the turn itself (lanes-turns). It
// returns at once where none waits. The wait is on the turn's record and on the waiters' ends, never on a clock.
func YieldTurn(name string, release func() error) error {
	waiters, err := turnWaiters(name)
	if err != nil || len(waiters) == 0 {
		return errors.Join(err, release())
	}
	self, passed := uint32(os.Getpid()), make(chan struct{})
	pass := sync.OnceFunc(func() { close(passed) })
	stop, err := watchTurn(name, func(record turnRecord) {
		if record.Process != self {
			pass()
		}
	})
	if err != nil {
		return errors.Join(err, release())
	}
	defer stop()
	var ended sync.WaitGroup
	for _, pid := range waiters {
		if waiter, found := os.FindProcess(int(pid)); found == nil {
			ended.Go(func() { _, _ = waiter.Wait() })
		}
	}
	go func() { ended.Wait(); pass() }()
	if err := release(); err != nil {
		return err
	}
	if record, held, err := turnHolder(name); err == nil && held && record.Process != self {
		pass()
	}
	<-passed
	return nil
}

// TurnCount records, where a waiter reads it, how far this process has come in the work it holds its turns for, done
// of total, and when it got there, as a render does where it says "drew n/N": a waiter says its holder's count, so a
// reader tells a holder that advances from one that does not by the holder's own count, not a clock.
func TurnCount(done, total int) error {
	turnMu.Lock()
	defer turnMu.Unlock()
	var result error
	for _, hold := range ownTurns() {
		result = errors.Join(result, hold.count(done, total))
	}
	return result
}

// ownTurns are the turns this process holds in its own place, by name, with turnMu held.
func ownTurns() map[string]*turnHold {
	own := map[string]*turnHold{}
	for name, hold := range turns {
		select {
		case <-hold.taken:
			if hold.err == nil && hold.give != nil {
				own[name] = hold
			}
		default:
		}
	}
	return own
}

// takePlace gives this process a place on the turn: its holder's, where the holder started it, and nothing to give up
// or count, else its own, waited for, how it is given up and how its count is recorded. The holder named to it
// (TurnEnvironment) holds the turn where it holds the turn's record open (turnUsers), whatever the record reads while
// its count is rewritten (TurnCount): a child that read a record cut short by a rewrite took its holder for gone and
// waited for the turn its own parent held (SQA-394).
func takePlace(name, purpose string, output io.Writer) (give func() error, count func(done, total int) error, err error) {
	if holder, riding := heldTurns()[name]; riding {
		if users, err := turnUsers(name); err != nil {
			return nil, nil, err
		} else if slices.Contains(users, holder) {
			return nil, nil, nil
		}
	}
	started := time.Now()
	// The holder reads the mark (TurnWaited) while this process waits. It goes up before the turn is looked at, so a
	// process that takes the turn after a look that found it free still reads this one waiting, where an unmarked
	// waiter waited out the holder's whole work (SQA-261).
	unmark, err := markTurnWaiter(name)
	if err != nil {
		return nil, nil, err
	}
	busy, err := lookAtTurn(name)
	if err != nil {
		return nil, nil, errors.Join(err, unmark())
	}
	waiter, stopWatch := processLabel(uint32(os.Getpid())), func() {}
	if busy {
		record, held, err := turnHolder(name)
		if err != nil {
			return nil, nil, errors.Join(err, unmark())
		}
		fmt.Fprintf(output, "%s waits for the turn %s, held by %s\n", waiter, name, holderSaid(record, held))
		// While it waits, the waiter says each holder the turn passes to, where it had said the first alone and a
		// landing's line named a test the card had long since passed from (turn-holder/do).
		var said uint32
		if held {
			said = record.Process
		}
		if stopWatch, err = watchTurn(name, func(next turnRecord) {
			if next.Process != said {
				said = next.Process
				fmt.Fprintf(output, "%s waits for the turn %s, now held by %s\n", waiter, name, holderSaid(next, true))
			}
		}); err != nil {
			return nil, nil, errors.Join(err, unmark())
		}
	}
	leave, err := enterTurn(name)
	stopWatch()
	if err != nil {
		return nil, nil, errors.Join(err, unmark())
	}
	if err := unmark(); err != nil {
		return nil, nil, errors.Join(err, leave())
	}
	if busy {
		fmt.Fprintf(output, "%s took the turn %s after %s\n", processLabel(uint32(os.Getpid())), name, time.Since(started).Round(time.Millisecond))
	}
	record := turnRecord{Name: name, Process: uint32(os.Getpid()), Purpose: purpose, Since: time.Now()}
	rewrite, unrecord, err := recordTurn(name, record)
	if err != nil {
		return nil, nil, errors.Join(err, leave())
	}
	if err := setHeldTurn(name, true); err != nil {
		return nil, nil, errors.Join(err, unrecord(), leave())
	}
	count = func(done, total int) error {
		record.Done, record.Total, record.Reached = done, total, time.Now()
		return rewrite(record)
	}
	return func() error { return errors.Join(setHeldTurn(name, false), unrecord(), leave()) }, count, nil
}

// holderSaid is how a waiter names the turn's holder: its process, what it holds the turn for and since when, and how
// far it has counted, or that the holder has written no record.
func holderSaid(record turnRecord, held bool) string {
	if !held {
		return "a process that has written no record"
	}
	holder := fmt.Sprintf("%s for %s since %s", processLabel(record.Process), record.Purpose, record.Since.Format(time.DateTime))
	if record.Total > 0 {
		holder += fmt.Sprintf(", at %d/%d reached %s", record.Done, record.Total, record.Reached.Format(time.DateTime))
	}
	return holder
}

// lookAtTurn says whether another holds the turn now, for the line a waiter prints: ResourceTransactionHeld, which a
// test replaces to look as a race does.
var lookAtTurn = ResourceTransactionHeld

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
