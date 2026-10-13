package main

import (
	"flag"
	"fmt"
	"io"

	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
)

// censusRepeats is how many times checkpoint-families times each decode
// unless told: an odd count, so the middle reading is one reading, and
// enough of them that one disturbed reading is not it.
const censusRepeats = 5

// checkpointFamilies is overgodb checkpoint-families: it counts the
// checkpoint an open of the store loads, by projection and within each by
// record family, in bytes and in decode time. Every model load pays that
// decode, so the count says what each family of records costs a load. It
// opens the store to read it, prints the command and the store's sequence
// ahead of the census, and writes nothing to the store.
func checkpointFamilies(flags *flag.FlagSet, args []string, output io.Writer) error {
	repository := flags.String("repo", "", "OvergoDB store directory; empty resolves via the data-root contract (OVERGO_DATA_ROOT, local-models.json, or ./overgodb-store)")
	repeats := flags.Int("repeats", censusRepeats, "times each decode is timed; the middle reading is printed beside the least and the most")
	floor := flags.Float64("floor", 0, "percent of the checkpoint's bytes; a projection's families that each hold less are counted together in one line")
	if err := flags.Parse(args); err != nil {
		return err
	}
	root, err := dataroot.StoreRoot(*repository)
	if err != nil {
		return err
	}
	store, err := overgodb.OpenReadOnly(root)
	if err != nil {
		return err
	}
	head, sequence := store.Head()
	if err := store.Close(); err != nil {
		return err
	}
	fmt.Fprintf(output, "%s -repo %s -repeats %d -floor %v\nstore sequence %d head %s\n", flags.Name(), root, *repeats, *floor, sequence, head)
	return overgodb.WriteCheckpointCensus(output, root, *repeats, *floor)
}
