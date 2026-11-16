package evaluation

import (
	"errors"
	"fmt"
	"math"
)

// ScreenCase is one prompt and seed of a media screen: how far a candidate path's output lies from the reference path's
// at the same prompt and seed, and how far the reference path's output at another seed of the prompt lies from it, each
// one minus the cosine of the two outputs' image embeddings.
type ScreenCase struct {
	Name      string  `json:"name"`
	Candidate float64 `json:"candidate"`
	Reseeded  float64 `json:"reseeded"`
}

// Farther is whether the candidate lies farther from the reference than another seed of the prompt does.
func (c ScreenCase) Farther() bool { return c.Candidate > c.Reseeded }

// Screen is a diagnostic beside a candidate path's admission: each case, and how many lie farther from the reference
// than another seed of the prompt does, nearer, or as near. An embedding's distance establishes no loss of what a prompt
// asked (SQA-342), so the screen refuses no candidate and a person reads its cases.
type Screen struct {
	Cases   []ScreenCase `json:"cases"`
	Farther uint64       `json:"farther"`
	Nearer  uint64       `json:"nearer"`
	Ties    uint64       `json:"ties"`
}

// ScreenCandidate screens a candidate path by its cases.
func ScreenCandidate(cases []ScreenCase) (Screen, error) {
	if len(cases) == 0 {
		return Screen{}, errors.New("evaluation: a screen with no case says nothing")
	}
	screen := Screen{Cases: cases}
	for _, screened := range cases {
		for _, distance := range []float64{screened.Candidate, screened.Reseeded} {
			if math.IsNaN(distance) || math.IsInf(distance, 0) {
				return Screen{}, fmt.Errorf("evaluation: screen case %s has a distance that is not finite", screened.Name)
			}
		}
		switch {
		case screened.Farther():
			screen.Farther++
		case screened.Candidate < screened.Reseeded:
			screen.Nearer++
		default:
			screen.Ties++
		}
	}
	return screen, nil
}

// Says is the screen as a person reads it: a diagnostic, with its counts.
func (s Screen) Says() string {
	return fmt.Sprintf("a diagnostic, refusing no candidate: the candidate lies farther from the reference than another seed in %d of %d cases, nearer in %d, as near in %d",
		s.Farther, len(s.Cases), s.Nearer, s.Ties)
}
