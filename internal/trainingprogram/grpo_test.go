package trainingprogram

import (
	"math"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/hostoptimizer"
)

// TestGroupIsWeightedByOrderAlone holds the reward objective to its one rule:
// a group's answers are weighed by the order of their rewards and by nothing
// else of them.
//
// The weights. Each answer's gradient is the share of the others that beat
// it less the share it beats, over the group's size: the table's values, to
// the bit. They sum to nothing; equal rewards count for neither side; and a
// step down the gradients raises the best answer's log-probability and lowers
// the worst's. Rewards put through any strictly increasing change, a shift
// and a stretch, an exponential, a cube, give the same gradients to the bit,
// negative rewards and positive alike. A group whose answers all stand
// together gives zeros, and no negative zero among them. One answer is no
// group, and a reward that is not a number is refused.
//
// The instruction objective's ordering. More rules passed stands higher;
// among answers that pass every rule the shorter stands higher; and answers
// that fail the same number of rules stand together whatever their length, so
// that their gradients are those of a tie.
//
// The rule's place. A session plan for the reward objective takes no
// objective scale and says why, where a preference session requires one; and
// the rule is part of the identity of a program compiled for the objective,
// which is not the identity the same program had before it named a rule: a
// checkpoint trained under the scaled rewards is another program's.
func TestGroupIsWeightedByOrderAlone(t *testing.T) {
	t.Parallel()
	for name, group := range map[string]struct {
		rewards, want []float64
	}{
		"three in order":      {[]float64{-1, 0, 1}, []float64{1.0 / 3, 0, -1.0 / 3}},
		"two tied below one":  {[]float64{1, 1, 5}, []float64{1.0 / 2 / 3, 1.0 / 2 / 3, -1.0 / 3}},
		"a pair":              {[]float64{5, 1}, []float64{-1.0 / 2, 1.0 / 2}},
		"a tie in the middle": {[]float64{0, 1, 1, 3}, []float64{1.0 / 4, 0, 0, -1.0 / 4}},
		"all together":        {[]float64{2, 2, 2}, []float64{0, 0, 0}},
	} {
		got, err := OrderGradients(group.rewards)
		if err != nil || !slices.Equal(got, group.want) || slices.ContainsFunc(got, math.Signbit) && name == "all together" {
			t.Fatalf("%s: rewards %v give gradients %v (%v), want %v", name, group.rewards, got, err, group.want)
		}
		var sum float64
		for _, gradient := range got {
			sum += gradient
		}
		best, worst := slices.Index(group.rewards, slices.Max(group.rewards)), slices.Index(group.rewards, slices.Min(group.rewards))
		if math.Abs(sum) > 1e-15 || best != worst && (got[best] >= 0 || got[worst] <= 0) {
			t.Fatalf("%s: gradients %v sum to %v with the best answer's %v and the worst's %v; want nothing, a fall for the best to descend to and a rise for the worst", name, got, sum, got[best], got[worst])
		}
		for change, increasing := range map[string]func(float64) float64{
			"a shift and a stretch": func(reward float64) float64 { return 3*reward + 7 },
			"an exponential":        math.Exp,
			"a cube":                func(reward float64) float64 { return reward * reward * reward },
		} {
			changed := make([]float64, len(group.rewards))
			for index, reward := range group.rewards {
				changed[index] = increasing(reward)
			}
			if again, err := OrderGradients(changed); err != nil || !slices.Equal(again, got) {
				t.Fatalf("%s under %s: gradients %v (%v), want the same %v", name, change, again, err, got)
			}
		}
	}
	for name, rewards := range map[string][]float64{"one answer": {1}, "no answer": nil, "a reward that is no number": {1, math.NaN()}, "a reward without bound": {1, math.Inf(1)}} {
		if _, err := OrderGradients(rewards); err == nil {
			t.Errorf("%s was weighed", name)
		}
	}

	// The instruction ordering: answers to a prompt of three rules, as rules passed and tokens generated.
	const rules = 3
	passes := func(passed, tokens uint64) float64 { return instructionOrder(passed, rules, tokens) }
	for name, order := range map[string]struct{ higher, lower float64 }{
		"every rule passed over two":          {passes(3, 900), passes(2, 10)},
		"two rules passed over one":           {passes(2, 900), passes(1, 10)},
		"every rule passed, shorter":          {passes(3, 40), passes(3, 41)},
		"every rule passed, long, over fails": {passes(3, 1<<40), passes(2, 1)},
	} {
		if !(order.higher > order.lower) {
			t.Errorf("%s: %v does not stand above %v", name, order.higher, order.lower)
		}
	}
	if short, long := passes(2, 5), passes(2, 500); short != long {
		t.Errorf("two answers that fail one rule stand at %v and %v by their lengths, want them together", short, long)
	}
	tied, err := OrderGradients([]float64{passes(1, 5), passes(1, 500), passes(3, 60), passes(3, 20)})
	if err != nil || !slices.Equal(tied, []float64{1.0 / 2 / 3, 1.0 / 2 / 3, -1.0 / 3 / 4, -1.0 / 4}) {
		t.Fatalf("two failures of equal count and two passes of unequal length give %v (%v)", tied, err)
	}

	// The rule's place: no scale in a session, and the program's identity.
	spec := SessionSpec{
		Objective: ObjectiveGRPO, DatasetUnits: 1, MaximumSequence: 128,
		Optimizer: hostoptimizer.Config{BaseLearningRate: 0.01, Momentum: 0.9, Schedule: hostoptimizer.ScheduleConstant},
	}
	if _, err := CompileTrainingSessionPlan(spec); err != nil {
		t.Fatal(err)
	}
	spec.ObjectiveScale = 1
	if _, err := CompileTrainingSessionPlan(spec); err == nil || !strings.Contains(err.Error(), "takes no objective scale") {
		t.Fatalf("a reward session with an objective scale = %v, want it refused and told why", err)
	}
	spec.Objective, spec.ObjectiveScale = ObjectiveDPO, 0
	if _, err := CompileTrainingSessionPlan(spec); err == nil {
		t.Fatal("a preference session without an objective scale was planned")
	}
	plan, err := hostoptimizer.CompilePlan(1, []hostoptimizer.GroupSpec{{Name: "scalar", Start: 0, End: 1, Rows: 1, Cols: 1}})
	if err != nil {
		t.Fatal(err)
	}
	program, err := CompileTrainingProgram(ProgramSpec{
		Objective:  ObjectiveGRPO,
		Operators:  []OperatorSpec{{ID: "forward", Phase: PhaseForward}, {ID: "backward", Phase: PhaseBackward}, {ID: "optimize", Phase: PhaseOptimize}, {ID: "evaluate", Phase: PhaseEvaluate}},
		Parameters: []ParameterSpec{{Name: "scalar", Rows: 1, Cols: 1, Trainable: true}},
		Optimizer:  plan,
	})
	if err != nil {
		t.Fatal(err)
	}
	// The identity the same program had before the objective named its rule.
	before, err := artifact.JSONID(artifact.KindRecipe, struct {
		Objective   ObjectiveKind   `json:"objective"`
		Operators   []OperatorSpec  `json:"operators"`
		Parameters  []ParameterSpec `json:"parameters"`
		OptimizerID string          `json:"optimizer_id"`
	}{ObjectiveGRPO, program.Operators(), program.Parameters(), program.OptimizerIdentity()})
	if err != nil || !before.Valid() || before == program.ID() {
		t.Fatalf("the reward program is %s and was %s before it named its rule (%v), want another identity", program.ID(), before, err)
	}
}
