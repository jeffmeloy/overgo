package scorecard

import (
	"cmp"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"fmt"
	"math"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/scorecard/metric"
	"overgo/internal/strictjson"
)

// trainingTargetsJSON holds what training is to reach. A target is an input to the system: nothing here chooses a
// target or lowers one, and without one no gap asks for training at all.
//
//go:embed training_targets.json
var trainingTargetsJSON []byte

// TrainingTarget is one declared goal: the model, the held-out objective a run judges it on, the run that is to reach it,
// bounded by Steps, Sequence and MaxWall, and the gain, in the objective's own direction, a supported verdict must show.
// Order is the seed the run draws its data by: a bounded run reads only its first windows, which in membership order are the
// first few articles, and that run gains less than any drawn run a target's gain is read from.
type TrainingTarget struct {
	Model     artifact.ID `json:"model"`
	Name      string      `json:"name"`
	Alias     string      `json:"alias"`
	Objective artifact.ID `json:"objective"`
	Steps     int         `json:"steps"`
	Sequence  int         `json:"sequence"`
	Order     uint64      `json:"order"`
	MaxWall   string      `json:"max_wall"`
	Gain      float64     `json:"gain"`
}

// trainingKeyPrefix begins the key of a declared target, which carries its model: gaps of a kind group by task, and two
// models' targets are two rows.
const trainingKeyPrefix = gapKeyPrefix + "-train-"

// key names the gap of a target.
func (target TrainingTarget) key() string {
	sum := sha256.Sum256(fmt.Appendf(nil, "%s|%s", target.Model, target.Objective))
	return trainingKeyPrefix + hex.EncodeToString(sum[:])[:regressionDigits]
}

// met reads a row's held-out quality against the target: a supported verdict on its objective that gained what it asks.
func (target TrainingTarget) met(row Row) bool {
	for _, cell := range row.Measured(metric.Quality) {
		if cell.Supported && cell.Objective == target.Objective && cell.Base != nil {
			if gain, _ := cmp.Or(cell.Direction, columnDirections[metric.Quality]).Advantage(cell.Value, *cell.Base); gain >= target.Gain {
				return true
			}
		}
	}
	return false
}

// parseTargets reads a targets document, refusing one a run could not be started from.
func parseTargets(data []byte) ([]TrainingTarget, error) {
	var document struct {
		Targets []TrainingTarget `json:"targets"`
	}
	if err := strictjson.DecodeBytes(data, &document); err != nil {
		return nil, fmt.Errorf("scorecard: training targets: %w", err)
	}
	seen := map[string]bool{}
	for _, target := range document.Targets {
		wall, err := time.ParseDuration(target.MaxWall)
		if target.Model.Kind() != artifact.KindModel || !target.Objective.Valid() || strings.TrimSpace(target.Name) == "" || strings.TrimSpace(target.Alias) == "" ||
			target.Steps <= 0 || target.Sequence <= 0 || target.Order == 0 || err != nil || wall <= 0 || !(target.Gain > 0) || math.IsInf(target.Gain, 0) || seen[target.key()] {
			return nil, fmt.Errorf("scorecard: the training target of %q on %s is repeated or incomplete", target.Name, target.Alias)
		}
		seen[target.key()] = true
	}
	return document.Targets, nil
}

// declaredTargets are the targets Gaps reads; a malformed file fails every package that reads the scorecard, as a test of it does.
var declaredTargets = func() []TrainingTarget {
	targets, err := parseTargets(trainingTargetsJSON)
	if err != nil {
		panic(err)
	}
	return targets
}()

// trainingGaps are the targets of the record's models no supported verdict meets: each a row until one does. It is what
// starts the loop's train arc, since the only other gap that asks for training fires on a model already trained and below
// its base. A model the record does not train has no row to read a verdict from, and files nothing.
func (record Record) trainingGaps(targets []TrainingTarget) []Gap {
	var gaps []Gap
	for _, target := range targets {
		trained, met := false, false
		for _, row := range record.Rows {
			if row.Model == target.Model && row.Task == recipe.TaskTraining {
				trained, met = true, met || target.met(row)
			}
		}
		if trained && !met {
			gaps = append(gaps, Gap{
				Key: target.key(), kind: gapTraining,
				Title: fmt.Sprintf("Train %s on %s to a supported gain of %g", target.Name, target.Alias, target.Gain),
				Detail: fmt.Sprintf("- go run ./cmd/train -objective %s -model <its directory> -out <a new directory> -steps %d -seq %d -order-seed %d -freeze-lexical -max-wall %s\n"+
					"- the row closes when a supported held-out verdict on that objective gains %g\n", target.Alias, target.Steps, target.Sequence, target.Order, target.MaxWall, target.Gain),
			})
		}
	}
	return gaps
}
