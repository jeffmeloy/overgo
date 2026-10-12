package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/modelintake"
	"overgo/internal/overgodb"
	"overgo/internal/pathidentity"
	"overgo/internal/trainingworkflow"
)

const (
	// pilotRecord is the file under -out a pilot's record is added to: what
	// keeps the record once the checkpoints it speaks of are removed.
	pilotRecord = "pilot-record.txt"
	// adoptedKept is the reason an adopted reward arm's checkpoints are kept
	// when the run gave none: they are what the adoption would be made from.
	adoptedKept = "the reward arm is adopted"
	// keptAgain opens the rule of a checkpoint a run keeps because an earlier
	// record under the same output kept it; the earlier rule follows. A rule
	// given for one run would otherwise hold for that run alone, and the run
	// after it would remove, with nothing listed first, what a person asked
	// to keep.
	keptAgain = "an earlier record under this output kept this checkpoint, so no later run removes it; the earlier rule: "
)

// The fields of a line trainingworkflow.RemoveCheckpoints writes for a path,
// between tabs: what was done, the bytes, the place, the path and the rule,
// which is the rest of the line.
const (
	receiptVerb = iota
	_
	_
	receiptPath
	receiptRule
	receiptFields
)

// runPilot runs the instruction pilot under a published acceptance
// (trainingworkflow.RunInstructionPilot) and records what it came to
// (recordPilot). Not adopting is the pilot's answer and no failure of the
// command. The answers it draws are bound to the commit it was built at, so
// it refuses a tree with uncommitted source.
func runPilot(ctx context.Context, store *overgodb.Store, acceptanceText, model, out, kept string, output io.Writer) error {
	acceptance, err := artifact.ParseID(acceptanceText)
	if err != nil || model == "" || out == "" {
		return fmt.Errorf("train: -pilot names an instruction acceptance, as training-objective -pilot wrote it, and needs -model, the base it was fixed over, and -out: %w", err)
	}
	code, err := modelintake.CleanRevision(ctx)
	if err != nil {
		return err
	}
	result, err := trainingworkflow.RunInstructionPilot(ctx, store, trainingworkflow.PilotRun{Acceptance: acceptance, Model: model, Out: out, Code: code, Device: denseDevice})
	if err != nil {
		return err
	}
	return recordPilot(ctx, store, acceptance, code, out, kept, result, output)
}

// recordPilot writes what a pilot run came to, and then removes the
// checkpoints it trained. The record is what the base passed of the
// selection groups, each arm's checkpoints with what the session that reached
// each did and what it passed of the selection groups, the checkpoint frozen
// for each arm, each comparison's verdict beside the confirmation and the
// charge it is recorded under, and whether the reward arm is adopted, with
// the reason when it is not.
//
// The record goes to output and is added to pilotRecord under out, which is
// synced before anything is removed: a session's trace is nowhere else, so
// the weights go only once the record of the run that made them is in a file
// the removal does not touch, and a record that cannot be written fails the
// command with every checkpoint standing. The checkpoints are then removed
// (trainingworkflow.RemoveCheckpoints), a line each after the record, unless
// kept gives a reason to keep them or the reward arm is adopted.
//
// A checkpoint an earlier record in the file kept stays kept, under the rule
// it was kept for (keptBefore, keptAgain): the file is read before this run
// adds to it, and one that cannot be read fails the command like one that
// cannot be written.
func recordPilot(ctx context.Context, store *overgodb.Store, acceptance artifact.ID, code, out, kept string, result trainingworkflow.PilotResult, output io.Writer) (err error) {
	path := filepath.Join(out, pilotRecord)
	earlier, err := os.ReadFile(path)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	file, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, clioptions.PrivateFileMode)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, file.Close()) }()
	var record strings.Builder
	var directories []string
	fmt.Fprintf(&record, "instruction pilot run, built at commit %s\nacceptance: %s\nbase: passed %d of %d selection groups\n", code, acceptance, result.BaseSelected, result.Selection)
	for _, arm := range result.Arms {
		for _, checkpoint := range arm.Checkpoints {
			trace := checkpoint.Trace
			directories = append(directories, checkpoint.Directory)
			fmt.Fprintf(&record, "%s after %d groups: passed %d of %d selection groups; its session sampled %d groups, stepped on %d and passed %d, sampled %d tokens and trained on %d, %d answers at the output limit, %.0f s sampling, %.0f s scoring, %.0f s stepping, shift %.4f\n",
				arm.Name, checkpoint.Groups, checkpoint.Selected, result.Selection, trace.Sampled, trace.Updated, trace.Skipped, trace.SampledTokens, trace.TrainedTokens, trace.Truncated,
				trace.Generation.Seconds(), trace.Scoring.Seconds(), trace.Update.Seconds(), trace.Shift)
		}
		frozen := arm.Checkpoints[arm.Frozen]
		fmt.Fprintf(&record, "%s: frozen after %d groups, %s\n", arm.Name, frozen.Groups, frozen.Directory)
	}
	for _, comparison := range result.Comparisons {
		fmt.Fprintf(&record, "comparison %q: %s; confirmation %s, charge %s, %d left\n", comparison.Name, comparison.Confirmation.Verdict, comparison.Confirmation.ID, comparison.Confirmation.Charge, comparison.Left)
	}
	fmt.Fprintf(&record, "adopted: %t %s\n", result.Adopted, result.Reason)
	both := io.MultiWriter(output, file)
	if _, err := io.WriteString(both, record.String()); err != nil {
		return err
	}
	if err := file.Sync(); err != nil {
		return err
	}
	if kept == "" && result.Adopted {
		kept = adoptedKept
	}
	var own []string
	for _, directory := range directories {
		rule, held := keptBefore(string(earlier), directory)
		if kept != "" || !held {
			own = append(own, directory)
			continue
		}
		// A rule that was already kept again is carried as it is, so it does not grow a run at a time.
		if !strings.HasPrefix(rule, keptAgain) {
			rule = keptAgain + rule
		}
		err = errors.Join(err, trainingworkflow.RemoveCheckpoints(ctx, store, both, rule, directory))
	}
	return errors.Join(err, trainingworkflow.RemoveCheckpoints(ctx, store, both, kept, own...))
}

// keptBefore reads the lines for directory that earlier runs added to the
// record file, and says whether the newest of them kept it, and under what
// rule. A newest line that says left or removed keeps nothing: a removal that
// failed is tried again, and a checkpoint trained again after its removal is
// the run's own. A line is for the directory when it names its path or, under
// another spelling, the folder that stands there, since the same output is
// not always written the same way. The rule is the rest of its line, so one
// that holds a tab is read whole.
func keptBefore(record, directory string) (rule string, kept bool) {
	for line := range strings.Lines(record) {
		fields := strings.SplitN(strings.TrimSuffix(line, "\n"), "\t", receiptFields)
		if len(fields) != receiptFields {
			continue
		}
		if fields[receiptPath] != directory {
			if same, err := pathidentity.Same(directory, fields[receiptPath]); err != nil || !same {
				continue
			}
		}
		rule, kept = fields[receiptRule], fields[receiptVerb] == "kept"
	}
	return rule, kept
}
