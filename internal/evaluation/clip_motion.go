package evaluation

import (
	"cmp"
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/runrecord"
)

// The metrics a suite of clips records of how its clips move, beside its judged share (ClipMotion).
const (
	// FrameSimilarityMetric is the mean over the suite's clips of each clip's mean neighbour similarity.
	FrameSimilarityMetric = "frame-similarity"
	// LowestFrameSimilarityMetric is the mean over the suite's clips of each clip's lowest neighbour similarity: a
	// seam between chunks or a flicker, which no question about a clip's content sees.
	LowestFrameSimilarityMetric = "lowest-frame-similarity"
	// BelowSourceMetric is the share of the clips with a source clip whose lowest neighbour similarity falls below
	// their source's: a step between frames larger than any the real clip the caption describes takes.
	BelowSourceMetric = "below-source-frame-similarity"
	// StillerThanSourcesMetric is the share of the suite's clips more alike frame to frame, by their mean neighbour
	// similarity, than the stillest of the suite's source clips: a clip frozen on one frame, which the judge's caption
	// questions pass (judge-calibration/sensitivity) and whose mean and lowest similarity read as the best a clip can
	// have, reads here as too still.
	StillerThanSourcesMetric = "stiller-than-sources"
	// minimumMotionFrames is the fewest frames a clip has a neighbour pair in.
	minimumMotionFrames = 2
)

// FrameSimilarity is how a clip moves: the cosine of each frame's image embedding with the next frame's, their mean,
// and the lowest.
type FrameSimilarity struct {
	Neighbours []float64 `json:"neighbours"`
	Mean       float64   `json:"mean"`
	Lowest     float64   `json:"lowest"`
}

// ClipMotion is how a suite's clips move, each by what its cases ask about (MotionKey), with the image encoder that
// embedded their frames, and the same measure over the source clip each one describes where the suite has one: the
// floor real motion sets.
type ClipMotion struct {
	Encoder string                     `json:"encoder"`
	Clips   map[string]FrameSimilarity `json:"clips"`
	Sources map[string]FrameSimilarity `json:"sources,omitempty"`
}

// MotionKey is what a case's clip is measured by: its group, the caption it was drawn from, or where it has none its
// own name, as an edited clip's one case is named after the clip it edits.
func (c GeneratedAnswerCase) MotionKey() string { return cmp.Or(c.Group, c.Name) }

// Cosine is the cosine of the angle between two embeddings of one width; NaN where they differ in width or either is
// empty.
func Cosine(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.NaN()
	}
	var across, alongA, alongB float64
	for index, value := range a {
		x, y := float64(value), float64(b[index])
		across, alongA, alongB = across+x*y, alongA+x*x, alongB+y*y
	}
	return across / math.Sqrt(alongA*alongB)
}

// NewFrameSimilarity measures a clip by its frames' embeddings, in order.
func NewFrameSimilarity(frames [][]float32) (FrameSimilarity, error) {
	if len(frames) < minimumMotionFrames {
		return FrameSimilarity{}, fmt.Errorf("evaluation: a clip of %d frames has no neighbours", len(frames))
	}
	measured := FrameSimilarity{Neighbours: make([]float64, len(frames)-1)}
	for index := range measured.Neighbours {
		similarity := Cosine(frames[index], frames[index+1])
		if math.IsNaN(similarity) || math.IsInf(similarity, 0) {
			return FrameSimilarity{}, fmt.Errorf("evaluation: frames %d and %d embed to vectors with no cosine", index, index+1)
		}
		measured.Neighbours[index] = similarity
		measured.Mean += similarity / float64(len(measured.Neighbours))
	}
	measured.Lowest = slices.Min(measured.Neighbours)
	return measured, nil
}

// validate refuses motion that names a clip no case asks about, a source for a clip it lacks, or no encoder.
func (m ClipMotion) validate(cases []GeneratedAnswerCase) error {
	if m.Encoder == "" || len(m.Clips) == 0 {
		return errors.New("evaluation: a suite's clip motion names its encoder and at least one clip")
	}
	for group, measured := range m.Clips {
		if !slices.ContainsFunc(cases, func(testCase GeneratedAnswerCase) bool { return testCase.MotionKey() == group }) {
			return fmt.Errorf("evaluation: the suite's clip motion measures %s, which no case asks about", group)
		}
		if len(measured.Neighbours) == 0 {
			return fmt.Errorf("evaluation: the clip motion of %s holds no neighbours", group)
		}
	}
	for group := range m.Sources {
		if _, held := m.Clips[group]; !held {
			return fmt.Errorf("evaluation: the suite's clip motion measures the source of %s and not its clip", group)
		}
	}
	return nil
}

// Metrics are the motion metrics a judged reading of the suite records.
func (m ClipMotion) Metrics() []runrecord.Metric {
	var mean, lowest float64
	var below, sourced int
	for group, measured := range m.Clips {
		mean += measured.Mean / float64(len(m.Clips))
		lowest += measured.Lowest / float64(len(m.Clips))
		if source, held := m.Sources[group]; held {
			sourced++
			if measured.Lowest < source.Lowest {
				below++
			}
		}
	}
	metrics := []runrecord.Metric{
		{Name: FrameSimilarityMetric, Value: mean, Direction: runrecord.DirectionMaximize},
		{Name: LowestFrameSimilarityMetric, Value: lowest, Direction: runrecord.DirectionMaximize},
	}
	if sourced > 0 {
		metrics = append(metrics, runrecord.Metric{Name: BelowSourceMetric, Value: float64(below) / float64(sourced), Direction: runrecord.DirectionMinimize})
	}
	// The stillest real clip the suite's captions describe sets how still a clip of them may be; any clip of the suite
	// beyond it, a composed caption's too, reads as too still.
	if len(m.Sources) > 0 {
		var ceiling float64
		for _, source := range m.Sources {
			ceiling = max(ceiling, source.Mean)
		}
		var still int
		for _, measured := range m.Clips {
			if measured.Mean > ceiling {
				still++
			}
		}
		metrics = append(metrics, runrecord.Metric{Name: StillerThanSourcesMetric, Value: float64(still) / float64(len(m.Clips)), Direction: runrecord.DirectionMinimize})
	}
	return metrics
}
