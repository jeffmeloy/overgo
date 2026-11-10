package main

import (
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
)

// The README's written parts stand between these pairs of lines; -readme writes what stands between them.
const (
	readmeSpeedBegin                = "<!-- speed table: go run ./cmd/validate -readme writes it from the newest scorecard record -->"
	readmeSpeedEnd                  = "<!-- end of the speed table -->"
	readmeServedBegin               = "<!-- served-model table: go run ./cmd/validate -readme writes it from the newest scorecard record -->"
	readmeServedEnd                 = "<!-- end of the served-model table -->"
	readmeTrainingBegin             = "<!-- training line: go run ./cmd/validate -readme writes it from the training specifications and the newest scorecard record -->"
	readmeTrainingEnd               = "<!-- end of the training line -->"
	readmeFile                      = "README.md"
	readmeMode          fs.FileMode = 0o644
)

// readmePart is one part of the README -readme writes: the lines it stands between and what it is written from.
type readmePart struct {
	begin, end string
	write      func(root string, newest scorecard.Record) (string, error)
}

// readmeParts are the parts -readme writes, each where its marker lines stand.
var readmeParts = []readmePart{
	{readmeServedBegin, readmeServedEnd, func(_ string, newest scorecard.Record) (string, error) {
		words, err := scorecard.DeclaredServedWords()
		if err != nil {
			return "", err
		}
		table, err := newest.ServedTable(words)
		return fmt.Sprintf("Read from the record `%s`.\n\n%s", strings.TrimSuffix(newest.Name, ".json"), table), err
	}},
	{readmeTrainingBegin, readmeTrainingEnd, trainingLine},
	{readmeSpeedBegin, readmeSpeedEnd, func(_ string, newest scorecard.Record) (string, error) {
		return newest.SpeedTable() + newest.GenerationTable(), nil
	}},
}

// writeReadme writes the README beneath root's written parts, each between its two marker lines, from the newest
// scorecard record in records and the training specifications beneath root. With check it writes nothing and refuses a
// README any part of which is not what they give, so a hand-typed figure, a cold one among them, cannot stand in it. A
// README holding none of the parts is refused.
func writeReadme(root, records string, check bool, output io.Writer) error {
	published, err := scorecard.Records(records)
	if err != nil {
		return err
	}
	if len(published) == 0 {
		return fmt.Errorf("validate: no scorecard record in %s to write the README's tables from", records)
	}
	newest := published[len(published)-1]
	path := filepath.Join(root, readmeFile)
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	text, written := string(data), string(data)
	var parts []string
	for _, part := range readmeParts {
		before, rest, opened := strings.Cut(written, part.begin)
		_, after, closed := strings.Cut(rest, part.end)
		if !opened || !closed {
			continue
		}
		inside, err := part.write(root, newest)
		if err != nil {
			return err
		}
		written = before + part.begin + "\n" + inside + part.end + after
		parts = append(parts, part.begin)
	}
	if len(parts) == 0 {
		return fmt.Errorf("validate: %s holds no speed table, served-model table or training line between their marker lines", path)
	}
	if check {
		if written != text {
			return fmt.Errorf("validate: %s's written tables are not what %s gives; write them with go run ./cmd/validate -readme", path, newest.Name)
		}
		return nil
	}
	if err := os.WriteFile(path, []byte(written), readmeMode); err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "%s: %d written parts written from %s\n", path, len(parts), newest.Name)
	return err
}

// tierWords say what each valid tier of training evidence establishes, strongest first; a tier's span of steps is
// said where its claims are bounded runs.
var tierWords = []struct {
	tier  runrecord.VerificationTier
	words string
}{
	{runrecord.TierCapabilityMeasured, "a held-out measure of what training changed"},
	{runrecord.TierExactGolden, "steps that match a reference implementation's exactly"},
	{runrecord.TierRealArtifactSmoke, "a bounded run on the real model, of %d to %d steps, which shows the route runs and not that training converges"},
	{"synthetic-differential", "a synthetic model's steps checked against a reference"},
}

// trainingLine is the README's line on training (trainingWords), from the training specifications beneath root.
func trainingLine(root string, newest scorecard.Record) (string, error) {
	verifications, _, err := loadTrainingVerifications(root)
	if err != nil {
		return "", err
	}
	return trainingWords(verifications, newest)
}

// trainingWords is the README's line on training: each model of verifications named under the strongest tier of its
// training evidence, and each held-out verdict the newest record holds, against its base. A recipe is never called
// proven by more than its tier says.
func trainingWords(verifications []runrecord.ModelVerification, newest scorecard.Record) (string, error) {
	strongest := map[string]runrecord.CapabilityClaim{}
	for _, row := range runrecord.VerificationMatrix(verifications) {
		for _, claim := range row.Capabilities {
			if current, seen := strongest[row.Name]; trainingCapability(claim.Capability) && (!seen || claim.Tier.StrongerThan(current.Tier)) {
				strongest[row.Name] = claim
			}
		}
	}
	var tiers []string
	for _, tier := range tierWords {
		var names []string
		fewest, most := uint64(0), uint64(0)
		for name, claim := range strongest {
			if claim.Tier != tier.tier {
				continue
			}
			names = append(names, name)
			if fewest == 0 || claim.SpanSteps < fewest {
				fewest = claim.SpanSteps
			}
			most = max(most, claim.SpanSteps)
		}
		if len(names) == 0 {
			continue
		}
		slices.SortFunc(names, func(left, right string) int { return strings.Compare(strings.ToLower(left), strings.ToLower(right)) })
		words := tier.words
		if strings.Contains(words, "%d") {
			words = fmt.Sprintf(words, fewest, most)
		}
		tiers = append(tiers, fmt.Sprintf("`%s`, %s, for %s", tier.tier, words, strings.Join(names, ", ")))
	}
	words, err := scorecard.DeclaredServedWords()
	if err != nil {
		return "", err
	}
	var verdicts []string
	for _, row := range newest.Rows {
		quality := row.Cells[metric.Quality]
		if row.Task != recipe.TaskTraining || row.Stale != "" || !quality.Measured() {
			continue
		}
		measure, named := words.MeasureOf(quality)
		if !named {
			return "", fmt.Errorf("validate: %s's training verdict is scored on %q, which internal/scorecard/served_models.json names no words for", row.Name, quality.Measure)
		}
		verdicts = append(verdicts, row.Name+", "+quality.Brief(measure))
	}
	line := "Training, by the strongest tier of each recipe's evidence in the [training report](docs/TRAINING_COMPATIBILITY.md): " + strings.Join(tiers, "; ") + "."
	if len(verdicts) != 0 {
		line += " Held-out verdicts in the record: " + strings.Join(verdicts, "; ") + "."
	}
	return line + "\n", nil
}
