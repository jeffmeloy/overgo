package main

import (
	"errors"
	"time"

	"overgo/internal/processmeasure"
)

// generationTiming keeps both generation paths on the counter used by prompt
// evaluation. A separate presence bit preserves a real zero first-token reading.
type generationTiming struct {
	elapsed    func() (uint64, error)
	firstToken uint64
	hasToken   bool
}

func startGenerationTiming() generationTiming {
	return generationTiming{elapsed: processmeasure.NewStopwatch().Elapsed}
}

func (t *generationTiming) token() error {
	if t.hasToken {
		return nil
	}
	first, err := t.elapsed()
	if err != nil {
		return err
	}
	t.firstToken, t.hasToken = first, true
	return nil
}

func (t generationTiming) finish() (total, ttft, decode time.Duration, err error) {
	wall, err := t.elapsed()
	if err != nil {
		return 0, 0, 0, err
	}
	first := wall
	if t.hasToken {
		first = t.firstToken
	}
	if first > wall {
		return 0, 0, 0, errors.New("benchmark: counter moved backwards after first token")
	}
	return time.Duration(wall), time.Duration(first), time.Duration(wall - first), nil
}
