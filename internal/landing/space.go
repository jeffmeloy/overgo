package landing

import (
	"context"
	"fmt"
)

const (
	// spaceClass names the stage that reads free space, and the class of its refusal.
	spaceClass = "space"
	// gibibyte is the unit free space is said in.
	gibibyte = 1 << 30
	// largestCheckWrites is what one run of the host stages of a commit's
	// checks was measured to write, in bytes: the greatest fall of the volume's
	// free space, sampled every two seconds, during one run of those stages by
	// hand (the gate's build, go vet with the tags modeltest and integration
	// over the whole tree, go test over the whole tree) in a checkout directory
	// never built before and with an empty private build cache, which is the
	// worst case for those stages. Measured on 2026-10-12 at commit beb6a2137
	// on this machine (Go 1.26.2, Windows 11); the cache held 2,210,006,016
	// bytes afterwards. Other sessions' go processes were running, so their
	// writes are inside the figure. Not included: the device, browser and cuda
	// test runs, and the gate's fix, generator and benchmark-build stages.
	// Measure again when the tree's package count or the Go version changes
	// much.
	largestCheckWrites uint64 = 3_648_241_664
	// measurementRunWrites is what one full measurement run was measured to
	// need free, in bytes: the greatest fall of the volume's free space,
	// sampled every two seconds, from the start of go run ./cmd/loop -improve
	// to its end. Measured on 2026-10-12 and 2026-10-13 at commit cc82ed097 on
	// this machine, over 4 h 49 m: 106 cells acquired again, the scorecard
	// published, the model tests run, nothing to fill. The fall is a dip and
	// not growth: free space ended 11,614,150,656 bytes below where it began,
	// and the dip came in one minute, when the training proof of the 9B model
	// wrote its checkpoint, about 80 GB, under the temporary directory and
	// removed it. For the run's first 38 minutes another session compiled and
	// ran tests, so its writes are inside both figures. Measure again when a
	// larger model gains a training proof, or when the proof stops writing a
	// checkpoint it removes: the floor then falls to the run's growth.
	measurementRunWrites uint64 = 93_108_281_344
	// freeSpaceFloor is the least free space a commit's checks may start with:
	// the sum, since a measurement run writes beside commits.
	freeSpaceFloor = largestCheckWrites + measurementRunWrites
)

// space reads, once and before any other stage, the free space of the volume
// that holds the commit's temporary directory, and refuses the run below the
// floor or when the read fails: a volume that fills fails every later stage
// for a reason no stage names. A read that failed is said as that, with no
// figure: the reader's zero is not a reading.
func (l *landing) space(context.Context) error {
	free, err := l.Free(l.scratch)
	if err != nil {
		return fmt.Errorf("the free space of the volume of %s could not be read, so nothing says it is above the floor of %d bytes: %w", l.scratch, freeSpaceFloor, err)
	}
	found := fmt.Sprintf("free %.1f GiB, floor %.1f GiB", float64(free)/gibibyte, float64(freeSpaceFloor)/gibibyte)
	if free < freeSpaceFloor {
		return fmt.Errorf("%s (%d and %d bytes; the floor allows %d for a run of a commit's checks and %d for a measurement run) on the volume of %s: that is below the floor; go run ./cmd/loop -leftovers lists what can be removed",
			found, free, freeSpaceFloor, largestCheckWrites, measurementRunWrites, l.scratch)
	}
	l.details[spaceClass] = found
	return nil
}
