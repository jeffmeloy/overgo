package jinja

import (
	"errors"
	"fmt"
	"math"
)

// Limits bound what one render may take. A chat template arrives inside a
// model file, so it is input nobody here wrote: without a bound, one line of
// it can loop for as long as it likes or build a value as large as memory.
// Both are counted, never timed, so a render that is refused is refused at the
// same place on any machine. A limit of zero bounds nothing.
type Limits struct {
	// Steps bounds the render's work: one for each statement it runs, each
	// expression it evaluates, and each item a loop walks or a function makes.
	Steps int
	// Bytes bounds what the render holds and reads: the text it writes out,
	// every string and list an operation builds on the way, and every value an
	// operation reads through to compare or search it.
	Bytes int
}

// errLimit is the error of a render that went past its limits, or of a
// template that nests deeper than maxNesting.
var errLimit = errors.New("jinja: the template went past its limits")

const (
	// maxNesting bounds how deep a template nests, as written and as it runs:
	// its brackets and its blocks inside one another, the operators of one
	// expression, and its macros calling one another. It is the depth Python
	// lets the code a template compiles to recurse to before it raises, and
	// no chat template comes near it; the engine walks each level on its own
	// stack, which a template would otherwise be free to exhaust.
	maxNesting = 1000
	// wordBytes is what one item of a list or one entry of a mapping holds
	// beside the value itself: an interface, on the machines overgo runs on.
	wordBytes = 16
	// valueBytes is what a list or a mapping holds before its first item. It
	// is charged for each one a template builds, so a template cannot nest
	// values deeper than its bytes pay for, and the walks over a value, which
	// follow its nesting on the stack, stay as shallow as the limit is small.
	valueBytes = 64
)

// scaled is count items of each bytes, or more than any limit when that is
// more than an integer holds: a size that wrapped would be charged as small.
func scaled(count, each int) int {
	if count > 0 && each > math.MaxInt/count {
		return math.MaxInt
	}
	return count * each
}

// budget is what is left of a render's limits, and how deep it stands.
type budget struct {
	steps, bytes, nesting int
}

// newBudget is the whole of the limits, with no bound where a limit is zero.
func newBudget(limits Limits) budget {
	unbounded := func(limit int) int {
		if limit <= 0 {
			return math.MaxInt
		}
		return limit
	}
	return budget{steps: unbounded(limits.Steps), bytes: unbounded(limits.Bytes)}
}

// work charges steps, and refuses a count the render cannot pay for before
// any of it is done.
func (b *budget) work(steps int) error {
	if steps < 0 || steps > b.steps {
		return fmt.Errorf("%w: its work", errLimit)
	}
	b.steps -= steps
	return nil
}

// hold charges bytes, and refuses a size the render cannot pay for before
// any of it is built.
func (b *budget) hold(bytes int) error {
	if bytes < 0 || bytes > b.bytes {
		return fmt.Errorf("%w: its memory", errLimit)
	}
	b.bytes -= bytes
	return nil
}

// enter charges one step for an expression and goes one level deeper; leave
// comes back out.
func (b *budget) enter() error {
	if b.nesting++; b.nesting > maxNesting {
		return fmt.Errorf("%w: it nests deeper than %d", errLimit, maxNesting)
	}
	return b.work(1)
}

func (b *budget) leave() { b.nesting-- }

// size is what a value an operation built holds itself: a string its length,
// a list or a mapping its own items, which are values charged when they were
// built or given.
func size(value any) int {
	switch made := value.(type) {
	case string:
		return len(made)
	case []any:
		return valueBytes + scaled(len(made), wordBytes)
	case map[string]any:
		return valueBytes + scaled(len(made), wordBytes)
	case *Dict:
		return valueBytes + scaled(len(made.order), wordBytes)
	}
	return 0
}

// made charges the value an operation built.
func (b *budget) made(value any) error { return b.hold(size(value)) }

// free gives back the bytes of a value nothing can hold any longer: the list
// a loop walked, and the value its own expression built for it to walk. A
// template that looks ahead from each message to the end builds a range a
// message, each gone when its loop ends; charged for good, they would count a
// long conversation's square against it.
func (b *budget) free(bytes int) {
	if bytes > 0 && bytes <= math.MaxInt-b.bytes {
		b.bytes += bytes
	}
}

// read charges the values an operation reads through, all of each: a
// comparison or a search follows a list into its items, so each is charged to
// its full depth, which is what the reading costs. A value that holds
// itself, as a namespace does once it is set as its own attribute, has no
// full depth: it is refused where it nests past maxNesting, like any other
// that does.
func (b *budget) read(values ...any) error { return b.readAt(0, values...) }

func (b *budget) readAt(depth int, values ...any) error {
	if depth > maxNesting {
		return fmt.Errorf("%w: a value nests deeper than %d, or holds itself", errLimit, maxNesting)
	}
	for _, value := range values {
		var err error
		switch read := value.(type) {
		case string:
			err = b.hold(len(read))
		case []any:
			err = errors.Join(b.hold(scaled(len(read), wordBytes)), b.readAt(depth+1, read...))
		case *Dict:
			for _, key := range read.order {
				if err = b.readAt(depth+1, key, read.vals[key]); err != nil {
					break
				}
			}
		case map[string]any:
			for key, item := range read {
				if err = b.readAt(depth+1, key, item); err != nil {
					break
				}
			}
		}
		if err != nil {
			return err
		}
	}
	return nil
}
