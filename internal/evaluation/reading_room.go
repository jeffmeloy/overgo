package evaluation

import (
	"fmt"
	"maps"
	"slices"

	"overgo/internal/tensorstats"
)

// ReadingRoom is what a reading against a reference could show at best, read from the reference's own answers
// before any arm is drawn (reading-power/do, the owner's direction of 2026-11-16): a preset can turn right only an
// answer the reference gives wrong, so the media holding one are the room a gain has, and a preset turning every wrong
// answer right and breaking none shows the best gain a reading of that reference can.
type ReadingRoom struct {
	// Media are the media the reference was asked about, and Room those it answers a question wrong about.
	Media int `json:"media"`
	Room  int `json:"room"`
	// BestTail is the grouped tail of a gain were a preset to turn every wrong answer right and break none, the
	// reading's statistic (presetVerdict), and PassMark the tail a gain must reach (falseGain).
	BestTail float64 `json:"best_tail"`
	PassMark float64 `json:"pass_mark"`
	// Needed is the fewest media a preset must turn an answer right in, breaking none, for a gain at the pass mark;
	// zero where the room holds too few.
	Needed int `json:"needed"`
}

// Possible says a gain could reach the pass mark against this reference.
func (room ReadingRoom) Possible() bool { return room.BestTail <= room.PassMark }

// ReadingRoomOf is the room a reading has against the reference whose cases' outcomes are outcomes (CaseOutcomes),
// each case's medium named by media (CaseMedia).
func ReadingRoomOf(outcomes map[string]bool, media map[string]string) (ReadingRoom, error) {
	wrong, asked := map[string]int64{}, map[string]bool{}
	for name, right := range outcomes {
		medium, named := media[name]
		if !named {
			return ReadingRoom{}, fmt.Errorf("evaluation: case %s asks about no medium the suite names, so the room has no unit", name)
		}
		asked[medium] = true
		if !right {
			wrong[medium]++
		}
	}
	room := ReadingRoom{Media: len(asked), Room: len(wrong), PassMark: falseGain,
		BestTail: tensorstats.GroupedSignTail(slices.Collect(maps.Values(wrong)))}
	for needed := 1; needed <= room.Room; needed++ {
		if tensorstats.GroupedSignTail(slices.Repeat([]int64{1}, needed)) <= falseGain {
			room.Needed = needed
			break
		}
	}
	return room, nil
}
