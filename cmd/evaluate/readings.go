package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/processcontrol"
	"overgo/internal/strictjson"
)

// readingsCampaign declares a preset-readings campaign (guidance-ops/preset-readings): each model's reference and each
// of its presets drawn at the size a person asks for, judged once, and each preset read against the reference by
// -admit-preset. Its records and verdicts are written beside its logs in Out, so a campaign run again resumes: a render
// resumes from its manifest, an arm whose record is written is not judged again, and a verdict is not read again.
type readingsCampaign struct {
	Out        string          `json:"out"`
	Repository string          `json:"repository"`
	Catalog    string          `json:"catalog"`
	Judge      string          `json:"judge"`
	Projector  string          `json:"projector"`
	Models     []readingsModel `json:"models"`
}

// readingsModel is one model of a campaign: its reference drawn into Reference, and each preset into <Arms>-<preset>.
type readingsModel struct {
	Name      string   `json:"name"`
	Path      string   `json:"path"`
	Task      string   `json:"task"`
	Source    string   `json:"source"`
	Cases     int      `json:"cases"`
	Reference string   `json:"reference"`
	Arms      string   `json:"arms"`
	Presets   []string `json:"presets"`
	// Negative is, by preset, the negative prompt its arm asks each case with (-render-negative).
	Negative map[string]string `json:"negative,omitempty"`
	// Pairwise reads each preset's arm of a picture model pairwise against the reference beside its questions
	// (-pairwise-preset), which pairwise-reading/sensitivity showed the judge to see.
	Pairwise bool `json:"pairwise,omitempty"`
}

// readingsRender is the render flag that draws a task's suite from its source.
var readingsRender = map[string]string{"image-gen": "-render-tifa", "video-gen": "-render-vidgen"}

// readingsRunner runs this build of evaluate with args, its output to stdout and stderr.
type readingsRunner func(ctx context.Context, args []string, stdout, stderr io.Writer) error

// readingArm is one drawn and judged suite of a campaign: a model's reference (no preset), one of its presets, or a
// preset's pairwise reading, which composes the reference's and the preset's renders it pairs where the others draw.
type readingArm struct {
	model         readingsModel
	name, preset  string
	media, record string
	pairs         *[2]string
	// walls is how long the arm took to draw and judge, where this campaign timed it.
	walls time.Duration
}

func (arm readingArm) suite() string { return arm.media + "-suite.json" }

// readings runs a campaign, its log in Out/campaign.log.
type readings struct {
	campaign readingsCampaign
	run      readingsRunner
	commit   string
	now      func() time.Time
	log      io.Writer
	// records and verdicts are what the campaign has written, by arm.
	records, verdicts map[string]string
}

const (
	readingsRecords  = "records.txt"
	readingsVerdicts = "verdicts.txt"
	// evaluatedLine opens the line a judged suite prints with the record it committed (evaluateSuiteFile).
	evaluatedLine = "evaluation "
	evaluatedOf   = " of "
)

func readReadingsCampaign(path string) (readingsCampaign, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return readingsCampaign{}, err
	}
	var campaign readingsCampaign
	if err := strictjson.DecodeBytes(data, &campaign); err != nil {
		return readingsCampaign{}, fmt.Errorf("evaluate: decode readings campaign %s: %w", path, err)
	}
	if campaign.Out == "" || campaign.Repository == "" || campaign.Catalog == "" || campaign.Judge == "" || campaign.Projector == "" || len(campaign.Models) == 0 {
		return readingsCampaign{}, fmt.Errorf("evaluate: readings campaign %s names no out, repository, catalog, judge, projector or models", path)
	}
	names := map[string]bool{}
	for _, model := range campaign.Models {
		if _, known := readingsRender[model.Task]; !known {
			return readingsCampaign{}, fmt.Errorf("evaluate: readings campaign %s: model %q has task %q, which draws no suite (image-gen or video-gen)", path, model.Name, model.Task)
		}
		if model.Name == "" || strings.ContainsAny(model.Name, " \t") || names[model.Name] || model.Path == "" || model.Source == "" || model.Cases <= 0 || model.Reference == "" || model.Arms == "" {
			return readingsCampaign{}, fmt.Errorf("evaluate: readings campaign %s: model %q needs a name of its own with no space, a path, a source, cases, a reference and arms", path, model.Name)
		}
		for preset := range model.Negative {
			if !slices.Contains(model.Presets, preset) {
				return readingsCampaign{}, fmt.Errorf("evaluate: readings campaign %s: model %q names a negative prompt for %q, none of its presets", path, model.Name, preset)
			}
		}
		if model.Pairwise && model.Task != "image-gen" {
			return readingsCampaign{}, fmt.Errorf("evaluate: readings campaign %s: model %q is read pairwise, which sets pictures side by side, and draws %s", path, model.Name, model.Task)
		}
		names[model.Name] = true
	}
	return campaign, nil
}

// runReadings runs the campaign at path with this build, whose commit each judged run names as its producer: a model
// whose reference cannot be drawn or judged is passed over for the next, as is one whose reference leaves no gain able
// to show (room), its arms undrawn, a preset whose arm cannot be drawn or judged for the next preset, and the error
// names every arm that failed once the rest are read.
func runReadings(ctx context.Context, path string, run readingsRunner, commit string, now func() time.Time) (result error) {
	campaign, err := readReadingsCampaign(path)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(campaign.Out, 0o755); err != nil {
		return err
	}
	logFile, err := os.OpenFile(filepath.Join(campaign.Out, "campaign.log"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer func() { result = errors.Join(result, logFile.Close()) }()
	r := &readings{campaign: campaign, run: run, commit: commit, now: now, log: io.MultiWriter(os.Stdout, logFile)}
	if r.records, err = readArmLines(filepath.Join(campaign.Out, readingsRecords)); err != nil {
		return err
	}
	if r.verdicts, err = readArmLines(filepath.Join(campaign.Out, readingsVerdicts)); err != nil {
		return err
	}
	r.say("campaign at %s", commit)
	var failed []error
	for _, model := range campaign.Models {
		reference := readingArm{model: model, name: model.Name + "-reference", media: model.Reference}
		if err := r.arm(ctx, &reference); err != nil {
			failed = append(failed, err)
			continue
		}
		if room, err := r.room(ctx, reference); err != nil || !room.Possible() {
			if err != nil {
				failed = append(failed, err)
			}
			continue
		}
		for _, preset := range model.Presets {
			arm := readingArm{model: model, name: model.Name + "-" + preset, preset: preset, media: model.Arms + "-" + preset}
			read := r.arm(ctx, &arm)
			if read == nil {
				read = r.admit(ctx, reference, arm)
			}
			if read == nil && model.Pairwise {
				read = r.pairwise(ctx, reference, arm)
			}
			if read != nil {
				failed = append(failed, read)
			}
		}
	}
	r.say("campaign ended, %d arms failed", len(failed))
	return errors.Join(failed...)
}

// readArmLines reads a campaign file whose each line opens with an arm's name, keyed by it.
func readArmLines(path string) (map[string]string, error) {
	lines := map[string]string{}
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return lines, nil
	}
	if err != nil {
		return nil, err
	}
	for line := range strings.Lines(string(data)) {
		if name, rest, found := strings.Cut(strings.TrimSpace(line), " "); found {
			lines[name] = rest
		}
	}
	return lines, nil
}

func (r *readings) say(format string, args ...any) {
	fmt.Fprintf(r.log, "%s %s\n", r.now().Format(time.DateTime), fmt.Sprintf(format, args...))
}

// append writes an arm's line to a campaign file and keeps it in lines.
func (r *readings) append(file string, lines map[string]string, name, rest string) error {
	handle, err := os.OpenFile(filepath.Join(r.campaign.Out, file), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(handle, "%s %s\n", name, rest)
	if err = errors.Join(err, handle.Close()); err == nil {
		lines[name] = rest
	}
	return err
}

// arm draws an arm's suite where it is not drawn, and judges it where it is not judged, setting its record.
func (r *readings) arm(ctx context.Context, arm *readingArm) error {
	if recorded, done := r.records[arm.name]; done {
		fields := strings.Fields(recorded)
		arm.record, arm.walls = fields[0], recordedWalls(fields)
		r.say("%s: judged already, %s", arm.name, arm.record)
		return nil
	}
	began := r.now()
	if err := r.draw(ctx, *arm); err != nil {
		return r.failed(arm.name, err)
	}
	record, err := r.judge(ctx, *arm)
	if err != nil {
		return r.failed(arm.name, err)
	}
	arm.walls = r.now().Sub(began)
	if err := r.append(readingsRecords, r.records, arm.name, fmt.Sprintf("%s %s %s%d", record, arm.media, wallsField, int64(arm.walls.Seconds()))); err != nil {
		return r.failed(arm.name, err)
	}
	arm.record = record
	r.say("%s: judged, %s", arm.name, record)
	return nil
}

// wallsField opens the field of an arm's records line that holds, in seconds, how long its draw and judging took.
const wallsField = "walls="

// recordedWalls is how long an arm's draw and judging took, as its records line holds it; zero where it holds none,
// a line written before the campaign timed its arms.
func recordedWalls(fields []string) time.Duration {
	for _, field := range fields {
		if seconds, held := strings.CutPrefix(field, wallsField); held {
			if count, err := strconv.ParseInt(seconds, 10, 64); err == nil {
				return time.Duration(count) * time.Second
			}
		}
	}
	return 0
}

// room reads, before any of a model's arms is drawn, the room a reading against its reference has, from the reference's
// own answers (evaluation.ReadingRoom), and says it with the card hours the arms take: where no gain can show, the
// campaign draws none of them (reading-power/do).
func (r *readings) room(ctx context.Context, reference readingArm) (evaluation.ReadingRoom, error) {
	var said bytes.Buffer
	if err := r.logged(ctx, reference.name, "room", []string{"-reading-room", reference.record, "-repo", r.campaign.Repository}, &said); err != nil {
		return evaluation.ReadingRoom{}, r.failed(reference.name, err)
	}
	var room evaluation.ReadingRoom
	if err := json.Unmarshal(bytes.TrimSpace(said.Bytes()), &room); err != nil {
		return evaluation.ReadingRoom{}, r.failed(reference.name, fmt.Errorf("the reading's room printed nothing to read: %w", err))
	}
	arms := len(reference.model.Presets)
	hours := "card hours unknown, its reference's draw and judging untimed"
	if reference.walls > 0 {
		hours = fmt.Sprintf("about %.1f card hours, each as its reference's %s", float64(arms)*reference.walls.Hours(), reference.walls.Round(time.Minute))
	}
	if !room.Possible() {
		r.say("%s: no gain can show: its reference answers a question wrong in %d of %d media, so the best a reading shows is a gain's tail of %.3g, over the pass mark %g; its %d arms are not drawn, %s",
			reference.model.Name, room.Room, room.Media, room.BestTail, room.PassMark, arms, hours)
	} else {
		r.say("%s: a gain shows only where a preset turns a wrong answer right in at least %d of the %d media its reference answers wrong in, breaking none (best tail %.3g); its %d arms take %s",
			reference.model.Name, room.Needed, room.Room, room.BestTail, arms, hours)
	}
	return room, nil
}

func (r *readings) failed(name string, err error) error {
	err = fmt.Errorf("%s: %w", name, err)
	r.say("%v", err)
	return err
}

// logged runs args with their output to the campaign's log file of the arm and step, and to into where it is not nil.
func (r *readings) logged(ctx context.Context, name, step string, args []string, into io.Writer) error {
	path := filepath.Join(r.campaign.Out, name+"-"+step+".log")
	handle, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	stdout := io.Writer(handle)
	if into != nil {
		stdout = io.MultiWriter(handle, into)
	}
	err = r.run(ctx, args, stdout, handle)
	if err = errors.Join(err, handle.Close()); err != nil {
		return fmt.Errorf("%s failed (%w); see %s", step, err, path)
	}
	return nil
}

// draw draws an arm's suite where it is not drawn; a pairwise arm, cheap to compose and drawing nothing, is composed
// each time it is not yet judged, so a suite written by an earlier build is not judged by this one.
func (r *readings) draw(ctx context.Context, arm readingArm) error {
	if _, err := os.Stat(arm.suite()); err == nil && arm.pairs == nil {
		return nil
	}
	r.say("%s: drawing into %s", arm.name, arm.media)
	args := []string{readingsRender[arm.model.Task], arm.model.Source, "-model-path", arm.model.Path, "-cases", strconv.Itoa(arm.model.Cases),
		"-judge", r.campaign.Judge, "-suite-out", arm.suite(), "-media-out", arm.media, "-repo", r.campaign.Repository}
	switch negative := arm.model.Negative[arm.preset]; {
	case arm.pairs != nil:
		args = []string{"-pairwise-preset", arm.pairs[1], "-reference-render", arm.pairs[0], "-suite-out", arm.suite(), "-media-out", arm.media}
	case negative != "":
		args = append(args, "-render-negative", negative)
		fallthrough
	case arm.preset != "":
		args = append(args, "-render-preset", arm.preset)
	}
	if err := r.logged(ctx, arm.name, "render", args, nil); err != nil {
		return err
	}
	if _, err := os.Stat(arm.suite()); err != nil {
		return fmt.Errorf("the render wrote no suite %s", arm.suite())
	}
	return nil
}

// judge reads an arm's suite with the campaign's judge in a manifest of its own, and gives the evaluation record the
// judged run printed for that suite: a record found as the store's newest would be another process's where one
// committed in between. A question suite's manifest names the generator, whose drawings its judge reads; a pairwise
// suite's names the judge itself with its projector, since no model drew its composites.
func (r *readings) judge(ctx context.Context, arm readingArm) (string, error) {
	r.say("%s: judging %s", arm.name, arm.suite())
	asked := modelRequest{Path: arm.model.Path, Task: arm.model.Task, Suites: []string{arm.suite()},
		Judge: r.campaign.Judge, Projector: r.campaign.Projector, MediaRoot: arm.media}
	if arm.pairs != nil {
		asked = modelRequest{Path: r.campaign.Judge, Suites: []string{arm.suite()}, Projector: r.campaign.Projector, MediaRoot: arm.media}
	}
	encoded, err := json.MarshalIndent(manifest{
		Repository: r.campaign.Repository, Catalog: r.campaign.Catalog, CodeCommit: r.commit, Models: []modelRequest{asked},
	}, "", "  ")
	if err != nil {
		return "", err
	}
	path := filepath.Join(r.campaign.Out, arm.name+"-manifest.json")
	if err := os.WriteFile(path, append(encoded, '\n'), 0o644); err != nil {
		return "", err
	}
	var said bytes.Buffer
	if err := r.logged(ctx, arm.name, "judge", []string{"-manifest", path}, &said); err != nil {
		return "", err
	}
	return evaluatedRecord(&said, arm.suite())
}

// evaluatedRecord is the evaluation record a judged run's output names for suite.
func evaluatedRecord(said io.Reader, suite string) (string, error) {
	lines := bufio.NewScanner(said)
	lines.Buffer(nil, 1<<20)
	for lines.Scan() {
		rest, evaluated := strings.CutPrefix(strings.TrimSpace(lines.Text()), evaluatedLine)
		record, of, named := strings.Cut(rest, evaluatedOf)
		if !evaluated || !named || filepath.Clean(of) != filepath.Clean(suite) {
			continue
		}
		id, err := artifact.ParseID(record)
		if err != nil || id.Kind() != artifact.KindEvaluation {
			return "", fmt.Errorf("the judged run named %q for %s, which is no evaluation record", record, suite)
		}
		return id.String(), nil
	}
	if err := lines.Err(); err != nil {
		return "", err
	}
	return "", fmt.Errorf("the judged run named no evaluation record of %s", suite)
}

// admit reads a preset's arm against its model's reference, once, and writes the verdict on one line.
func (r *readings) admit(ctx context.Context, reference, arm readingArm) error {
	if _, done := r.verdicts[arm.name]; done {
		return nil
	}
	var said bytes.Buffer
	if err := r.logged(ctx, arm.name, "admit", []string{"-admit-preset", arm.media, "-reference-render", reference.media,
		"-preset-evaluation", arm.record, "-reference-evaluation", reference.record, "-model-path", arm.model.Path,
		"-repo", r.campaign.Repository}, &said); err != nil {
		return r.failed(arm.name, err)
	}
	return r.verdict(arm.name, said.Bytes())
}

// pairwise reads a preset's arm pairwise against its model's reference, once: their pictures composed side by side in
// both orders, judged, and the verdict written as <arm>-pairwise's.
func (r *readings) pairwise(ctx context.Context, reference, arm readingArm) error {
	paired := readingArm{model: arm.model, name: arm.name + "-pairwise", preset: arm.preset, media: arm.media + "-pairwise", pairs: &[2]string{reference.media, arm.media}}
	if err := r.arm(ctx, &paired); err != nil {
		return err
	}
	if _, done := r.verdicts[paired.name]; done {
		return nil
	}
	var said bytes.Buffer
	if err := r.logged(ctx, paired.name, "admit", []string{"-pairwise-preset", arm.media, "-pairwise-evaluation", paired.record, "-repo", r.campaign.Repository}, &said); err != nil {
		return r.failed(paired.name, err)
	}
	return r.verdict(paired.name, said.Bytes())
}

// verdict writes the verdict a reading printed on the arm's one line.
func (r *readings) verdict(name string, said []byte) error {
	var verdict bytes.Buffer
	if err := json.Compact(&verdict, bytes.TrimSpace(said)); err != nil {
		return r.failed(name, fmt.Errorf("the reading printed no verdict: %w", err))
	}
	if err := r.append(readingsVerdicts, r.verdicts, name, verdict.String()); err != nil {
		return r.failed(name, err)
	}
	r.say("%s: %s", name, verdict.String())
	return nil
}

// processReadings runs executable, this build of evaluate, as a supervised process of its own for each step.
func processReadings(executable string) readingsRunner {
	return func(ctx context.Context, args []string, stdout, stderr io.Writer) error {
		receipt, err := processcontrol.Run(ctx, processcontrol.Command{Path: executable, Args: args, Stdout: stdout, Stderr: stderr})
		if err != nil {
			return err
		}
		if receipt.ExitCode != 0 {
			return fmt.Errorf("exit status %d", receipt.ExitCode)
		}
		return nil
	}
}
