package evaluation

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/tensorstats"
)

// ScreenCase is one prompt and seed of a media screen: how far a candidate path's output lies from the reference path's
// at the same prompt and seed, and how far the reference path's output at another seed of the prompt lies from it, each
// one minus the cosine of the two outputs' image embeddings.
type ScreenCase struct {
	Name      string  `json:"name"`
	Candidate float64 `json:"candidate"`
	Reseeded  float64 `json:"reseeded"`
}

// Lost is whether the candidate lies farther from the reference than another seed of the prompt does. Two seeds of a
// prompt are two outputs the prompt allows, so a candidate farther than that has lost what the prompt asked.
func (c ScreenCase) Lost() bool { return c.Candidate > c.Reseeded }

// Screen is a candidate path screened against its reference: each case, how many it lost and kept, and the exact sign
// test's chance of losing at least that many were a case as likely kept as lost, as a candidate that differs from its
// reference only as much as another seed would be. Passed is whether that chance stays above the level at which a
// held-out comparison accepts a false gain, so only a candidate the screen passes is judged.
type Screen struct {
	Cases  []ScreenCase `json:"cases"`
	Lost   uint64       `json:"lost"`
	Kept   uint64       `json:"kept"`
	Ties   uint64       `json:"ties"`
	Tail   float64      `json:"tail"`
	Level  float64      `json:"level"`
	Passed bool         `json:"passed"`
}

// ScreenCandidate screens a candidate path by its cases.
func ScreenCandidate(cases []ScreenCase) (Screen, error) {
	if len(cases) == 0 {
		return Screen{}, errors.New("evaluation: a screen with no case says nothing")
	}
	screen := Screen{Cases: cases, Level: falseGain}
	for _, screened := range cases {
		for _, distance := range []float64{screened.Candidate, screened.Reseeded} {
			if math.IsNaN(distance) || math.IsInf(distance, 0) {
				return Screen{}, fmt.Errorf("evaluation: screen case %s has a distance that is not finite", screened.Name)
			}
		}
		switch {
		case screened.Lost():
			screen.Lost++
		case screened.Candidate < screened.Reseeded:
			screen.Kept++
		default:
			screen.Ties++
		}
	}
	screen.Tail = tensorstats.ExactPairedBinary(screen.Lost, screen.Kept, screen.Ties).Tail
	screen.Passed = screen.Tail > screen.Level
	return screen, nil
}
