package trainingworkflow

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// The things that happen to a model's training recipes in a lifecycle.
const (
	// opSelect is a run under the recipe that succeeds: the recipe is verified,
	// and a candidate is activated and takes its model's one training slot.
	opSelect = "select"
	// opRetry runs the recipe of the run before again, and succeeds.
	opRetry = "retry"
	// opFail and opCancel are runs under the recipe that record no success.
	opFail   = "fail"
	opCancel = "cancel"
	// opRetire retires the selected recipe on failed proof.
	opRetire = "retire"
	// opReinstate activates a superseded recipe again, on an accepted decision
	// over fresh proof: the one door back for a recipe a decision refused.
	opReinstate = "reinstate"
	// opRollback returns the slot of a retired recipe to the recipe its
	// activation took the slot from, reinstating that one.
	opRollback = "rollback"
	// opRestart closes the store and opens it.
	opRestart = "restart"
	// opReplay sends the last lifecycle message again.
	opReplay = "replay"
)

const (
	// lifecycleRecipes is the number of training recipes one model has.
	lifecycleRecipes = 3
	// generatedLifecycles and lifecycleLength are the number of generated
	// sequences and the steps of each.
	generatedLifecycles = 64
	lifecycleLength     = 16
	// lifecycleSeed is the second word of every generated sequence's seed.
	lifecycleSeed = 20261009
	// noRecipe is the recipe of a step or a slot that names none.
	noRecipe = -1
)

// lifecycleStep is one thing that happens, and the recipe it happens to.
type lifecycleStep struct {
	op     string
	recipe int
}

func (s lifecycleStep) String() string {
	switch s.op {
	case opRetire, opRollback, opRestart, opReplay:
		return s.op
	}
	return s.op + " " + recipeName(s.recipe)
}

// recipeName names a model's recipes A, B and C, and the absence of one.
func recipeName(index int) string {
	if index == noRecipe {
		return "none"
	}
	return string(rune('A' + index))
}

// recipeRecord is what the reference keeps of one recipe: its state, whether a
// decision refused it, and the recipe its last activation took the slot from.
type recipeRecord struct {
	status  recipe.Status
	refused bool
	over    int
}

// lifecycleModel is the reference a store is compared with. It keeps each
// recipe's own record and, apart from them, which recipe the model has
// selected: nothing in it reads the selection to say whether a recipe may be
// used.
type lifecycleModel struct {
	recipes  []recipeRecord
	selected int
}

func newLifecycleModel() *lifecycleModel {
	model := &lifecycleModel{recipes: make([]recipeRecord, lifecycleRecipes), selected: noRecipe}
	for index := range model.recipes {
		model.recipes[index] = recipeRecord{status: recipe.StatusCandidate, over: noRecipe}
	}
	return model
}

// predecessor returns the recipe a rollback restores: the one the selected
// recipe's activation took the slot from, when the selected recipe is retired.
func (m *lifecycleModel) predecessor() int {
	if m.selected == noRecipe || !m.recipes[m.selected].refused {
		return noRecipe
	}
	return m.recipes[m.selected].over
}

// retires reports whether a retirement finds an active recipe to retire.
func (m *lifecycleModel) retires() bool {
	return m.selected != noRecipe && m.recipes[m.selected].status == recipe.StatusActive
}

// reinstates reports whether the recipe is one a reinstatement activates: a
// superseded recipe that does not hold the slot.
func (m *lifecycleModel) reinstates(index int) bool {
	return m.recipes[index].status == recipe.StatusSuperseded && m.selected != index
}

// take gives a recipe the slot, active. The recipe that held it loses the
// slot and nothing else: it is superseded and stays as eligible as it was, and
// one already retired keeps its refusal.
func (m *lifecycleModel) take(index int) {
	held := m.selected
	if held != noRecipe && m.recipes[held].status == recipe.StatusActive {
		m.recipes[held].status = recipe.StatusSuperseded
	}
	m.recipes[index], m.selected = recipeRecord{status: recipe.StatusActive, over: held}, index
}

// apply moves the reference by one step. It returns the one recipe whose
// eligibility the step may change, or noRecipe: a retirement changes the
// retired recipe's, a reinstatement or a rollback the reinstated one's, and no
// other step changes any.
func (m *lifecycleModel) apply(step lifecycleStep) int {
	switch step.op {
	case opSelect, opRetry:
		// A run under an active recipe verifies it again, and one under a
		// recipe that lost the slot verifies it and leaves the slot where it is.
		if record := m.recipes[step.recipe]; !record.refused && record.status == recipe.StatusCandidate {
			m.take(step.recipe)
		}
	case opRetire:
		if m.retires() {
			retired := &m.recipes[m.selected]
			retired.status, retired.refused = recipe.StatusSuperseded, true
			return m.selected
		}
	case opReinstate:
		if m.reinstates(step.recipe) {
			m.take(step.recipe)
			return step.recipe
		}
	case opRollback:
		if restored := m.predecessor(); restored != noRecipe {
			m.take(restored)
			return restored
		}
	}
	return noRecipe
}

// lifecycleReading is what a store says of a model's training recipes: each
// one's state, the refusal a run asking to train under it meets, and which the
// model has selected.
type lifecycleReading struct {
	statuses []recipe.Status
	refusals []error
	selected int
}

// differs says where a reading departs from the reference, or nothing.
func (m *lifecycleModel) differs(reading lifecycleReading) string {
	if reading.selected != m.selected {
		return fmt.Sprintf("the store has selected recipe %s and the reference %s", recipeName(reading.selected), recipeName(m.selected))
	}
	for index, record := range m.recipes {
		if reading.statuses[index] != record.status {
			return fmt.Sprintf("recipe %s is %q in the store and %q in the reference", recipeName(index), reading.statuses[index], record.status)
		}
		if refused := reading.refusals[index] != nil; refused != record.refused {
			return fmt.Sprintf("recipe %s: a run asking to train under it is refused %t (%v), and the reference refuses it %t",
				recipeName(index), refused, reading.refusals[index], record.refused)
		}
	}
	return ""
}

// recipeLifecycle drives one model's training recipes in a store through the
// doors a run, a retirement and a reinstatement use.
type recipeLifecycle struct {
	t           *testing.T
	directory   string
	store       *overgodb.Store
	model       artifact.ID
	environment artifact.ID
	recipes     []recipe.Definition
	proofs      uint64
	// last sends the last lifecycle message again.
	last func() error
}

func newRecipeLifecycle(t *testing.T) *recipeLifecycle {
	t.Helper()
	root, ctx := t.TempDir(), t.Context()
	lifecycle := &recipeLifecycle{t: t, directory: filepath.Join(root, "store")}
	lifecycle.open()
	t.Cleanup(func() {
		if err := lifecycle.store.Close(); err != nil {
			t.Error(err)
		}
	})
	modelPath := filepath.Join(root, "model.safetensors")
	if err := os.WriteFile(modelPath, []byte("weights"), 0o600); err != nil {
		t.Fatal(err)
	}
	for index := range lifecycleRecipes {
		id, err := BootstrapRecipe(ctx, lifecycle.store, modelPath, fmt.Appendf(nil, "the dataset of objective %d", index), TokenContract)
		if err != nil {
			t.Fatal(err)
		}
		definition, err := recipe.RequireDefinition(ctx, lifecycle.store, id)
		if err != nil {
			t.Fatal(err)
		}
		lifecycle.recipes, lifecycle.model = append(lifecycle.recipes, definition), definition.Model
	}
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: "eligibility", OS: "test", Arch: "test", Device: "host", Backend: "go", Driver: "test",
	})
	if err != nil {
		t.Fatal(err)
	}
	lifecycle.environment = environment.ID
	lifecycle.commit(environment.Batch("eligibility/environment"))
	return lifecycle
}

func (f *recipeLifecycle) open() {
	f.t.Helper()
	store, err := overgodb.Open(f.directory)
	if err != nil {
		f.t.Fatal(err)
	}
	f.store = store
}

func (f *recipeLifecycle) commit(batch artifact.Batch, err error) {
	f.t.Helper()
	if err == nil {
		_, err = artifact.CommitBatch(f.t.Context(), f.store, batch)
	}
	if err != nil {
		f.t.Fatal(err)
	}
}

// session is what a training run does around the training itself: it is
// admitted, asks for the recipe's authority, records how it ended, and on
// success verifies the recipe.
func (f *recipeLifecycle) session(index int, outcome error) {
	f.t.Helper()
	ctx, id := f.t.Context(), f.recipes[index].ID
	observer, err := NewObserver(f.store, true)
	if err != nil {
		f.t.Fatal(err)
	}
	if err := observer.Admit(ctx, f.model, id); err != nil {
		f.t.Fatal(err)
	}
	if _, err := ResolveSessionAuthority(ctx, f.store, f.model, id); err != nil {
		outcome = err
	}
	observation, err := observer.Finish(ctx, f.model, id, outcome, 1)
	if err != nil {
		f.t.Fatal(err)
	}
	if outcome != nil {
		return
	}
	if err := observer.Verify(ctx, id, observation); err != nil {
		f.t.Fatal(err)
	}
	activation, found, err := modelrecipe.ActiveRecord(ctx, f.store, f.model, recipe.TaskTraining)
	if err != nil || !found || activation.Definition.ID != id {
		return
	}
	verification := modelrecipe.Verification{Gate: activation.Verified.Gate.ID, Run: activation.Verified.Run.ID}
	f.last = func() error {
		return modelrecipe.ActivateCapability(ctx, f.store, activation.Definition, verification, recipe.EvidenceVerified, trainingVerificationReason)
	}
}

// proof publishes a verifier's gate and run over a recipe, with the outcome
// given.
func (f *recipeLifecycle) proof(id artifact.ID, outcome runrecord.Outcome) modelrecipe.Verification {
	f.t.Helper()
	f.proofs++
	step, failure := runrecord.StepSucceeded, ""
	if outcome == runrecord.OutcomeFailed {
		step, failure = runrecord.StepFailed, "exact-mismatch"
	}
	record, err := runrecord.NewGateRecord(id, f.environment, strings.Repeat("f0", 20), outcome, failure, f.proofs, []runrecord.GateStep{{
		Name: "verify", Phase: runrecord.PhaseValidate, Outcome: step, DurationNS: f.proofs,
	}})
	if err != nil {
		f.t.Fatal(err)
	}
	f.commit(record.Batch(fmt.Sprintf("eligibility/proof/%d", f.proofs)))
	return modelrecipe.Verification{Gate: record.Result.ID, Run: record.Run.ID}
}

// selected returns the recipe the model's training slot names.
func (f *recipeLifecycle) selected() (int, *artifact.ID) {
	f.t.Helper()
	id, bound, err := artifact.ResolveAlias(f.t.Context(), f.store, modelrecipe.ActiveAlias(f.model, recipe.TaskTraining))
	if err != nil {
		f.t.Fatal(err)
	}
	if !bound {
		return noRecipe, nil
	}
	for index, definition := range f.recipes {
		if definition.ID == id {
			return index, &id
		}
	}
	f.t.Fatalf("the model's training slot names %s, none of its recipes", id)
	return noRecipe, nil
}

// retire retires the selected recipe on proof that failed over it.
func (f *recipeLifecycle) retire() error {
	f.t.Helper()
	index, _ := f.selected()
	if index == noRecipe {
		return errors.New("no recipe is selected")
	}
	ctx, definition := f.t.Context(), f.recipes[index]
	failed := f.proof(definition.ID, runrecord.OutcomeFailed)
	f.last = func() error {
		return modelrecipe.RetireActiveCapability(ctx, f.store, definition, failed, "its proof failed when it was run again")
	}
	return f.last()
}

// reinstate activates a superseded recipe again on fresh proof, over whichever
// recipe holds the slot.
func (f *recipeLifecycle) reinstate(index int) error {
	f.t.Helper()
	ctx, definition := f.t.Context(), f.recipes[index]
	fresh := f.proof(definition.ID, runrecord.OutcomeSucceeded)
	_, held := f.selected()
	key := fmt.Sprintf("eligibility/reinstated/%d", f.proofs)
	f.last = func() error {
		_, _, err := modelrecipe.ActivateVerified(ctx, f.store, key, definition, fresh, recipe.EvidenceVerified, "reinstated on fresh proof", nil, held)
		return err
	}
	return f.last()
}

// rollback returns the slot to the recipe the retired holder's activation took
// it from, on fresh proof over the recipe named.
func (f *recipeLifecycle) rollback(restored int) error {
	f.t.Helper()
	ctx := f.t.Context()
	fresh := f.proof(f.recipes[restored].ID, runrecord.OutcomeSucceeded)
	f.last = func() error {
		return modelrecipe.RollbackActivation(ctx, f.store, f.model, recipe.TaskTraining, fresh, recipe.EvidenceVerified, "restored on fresh proof")
	}
	return f.last()
}

func (f *recipeLifecycle) read() lifecycleReading {
	f.t.Helper()
	ctx := f.t.Context()
	reading := lifecycleReading{statuses: make([]recipe.Status, len(f.recipes)), refusals: make([]error, len(f.recipes))}
	reading.selected, _ = f.selected()
	for index, definition := range f.recipes {
		status, _, err := modelrecipe.Status(ctx, f.store, definition.ID)
		if err != nil {
			f.t.Fatal(err)
		}
		reading.statuses[index] = status
		_, reading.refusals[index] = ResolveSessionAuthority(ctx, f.store, f.model, definition.ID)
	}
	return reading
}

// runLifecycle takes one model's recipes through steps in a store, and after
// each holds the store to two things: no step changed whether a recipe may be
// trained under but for the one recipe the step is about, and the store reads
// as the reference does.
func runLifecycle(t *testing.T, steps []lifecycleStep) {
	t.Helper()
	lifecycle, model := newRecipeLifecycle(t), newLifecycleModel()
	before := lifecycle.read()
	if difference := model.differs(before); difference != "" {
		t.Fatalf("before any step: %s", difference)
	}
	for index, step := range steps {
		done := steps[:index+1]
		switch step.op {
		case opSelect, opRetry:
			lifecycle.session(step.recipe, nil)
		case opFail:
			lifecycle.session(step.recipe, errors.New("the run failed"))
		case opCancel:
			lifecycle.session(step.recipe, context.Canceled)
		case opRetire:
			if err := lifecycle.retire(); (err == nil) != model.retires() {
				t.Fatalf("after %v: the retirement answered %v, and the reference retires a recipe: %t", done, err, model.retires())
			}
		case opReinstate:
			if model.reinstates(step.recipe) {
				if err := lifecycle.reinstate(step.recipe); err != nil {
					t.Fatalf("after %v: %v", done, err)
				}
			}
		case opRollback:
			// A rollback with nothing to restore is refused whatever proof it brings.
			restored := model.predecessor()
			if err := lifecycle.rollback(max(restored, 0)); (err == nil) != (restored != noRecipe) {
				t.Fatalf("after %v: the rollback answered %v, and the reference restores recipe %s", done, err, recipeName(restored))
			}
		case opRestart:
			if err := lifecycle.store.Close(); err != nil {
				t.Fatal(err)
			}
			lifecycle.open()
		case opReplay:
			if lifecycle.last != nil {
				t.Logf("after %v: the message sent again answered %v", done, lifecycle.last())
			}
		}
		subject := model.apply(step)
		after := lifecycle.read()
		for other := range after.refusals {
			if other != subject && (after.refusals[other] == nil) != (before.refusals[other] == nil) {
				t.Fatalf("after %v: the step changed whether a run may train under recipe %s, which it is not about: refused %v before and %v after",
					done, recipeName(other), before.refusals[other], after.refusals[other])
			}
		}
		if difference := model.differs(after); difference != "" {
			t.Fatalf("after %v: %s", done, difference)
		}
		before = after
	}
}

// generateLifecycle draws a sequence of steps. A retry is of the recipe the
// run before it trained under.
func generateLifecycle(random *rand.Rand) []lifecycleStep {
	ops := []string{
		opSelect, opSelect, opSelect, opSelect, opRetry, opFail, opCancel,
		opRetire, opRetire, opRetire, opReinstate, opReinstate, opRollback, opRollback, opRestart, opReplay,
	}
	steps, ran := make([]lifecycleStep, lifecycleLength), 0
	for index := range steps {
		step := lifecycleStep{op: ops[random.IntN(len(ops))], recipe: random.IntN(lifecycleRecipes)}
		switch step.op {
		case opRetry:
			step.recipe = ran
		case opSelect, opFail, opCancel:
			ran = step.recipe
		}
		steps[index] = step
	}
	return steps
}

// TestSelectionNeverRestoresEligibility holds the rule that changing which
// recipe a model has selected cannot change whether another recipe may be
// used. Whether a run may train under a recipe is what the recipe's own record
// says: a recipe that lost its model's one training slot to another recipe's
// activation may still be trained under, as runs of different objectives
// alternating on one model need, and a recipe retired on failed proof may not,
// whatever is selected after it, until an accepted decision over fresh proof
// activates it again.
//
// Two sequences are fixed. One is the sequence the rule was written for: A is
// selected and retired, B is selected, A is asked for and refused, and A is
// admitted once it is reinstated. The other is the alternation the superseded
// state was first admitted for. The rest are generated: runs that succeed,
// fail and are cancelled, retries, retirements, reinstatements, restarts of
// the store and messages sent twice, over three recipes of one model. Each is
// run against a real store through the doors a training run, a retirement and
// a reinstatement use, and after every step the store is held to a reference
// that keeps eligibility and selection apart.
func TestSelectionNeverRestoresEligibility(t *testing.T) {
	stubExecutableCodeCommit(t, strings.Repeat("e1", 20))
	const a, b = 0, 1
	t.Run("retired, another selected, asked for", func(t *testing.T) {
		runLifecycle(t, []lifecycleStep{
			{op: opSelect, recipe: a}, {op: opRetire}, {op: opSelect, recipe: b},
			{op: opFail, recipe: a}, {op: opReinstate, recipe: a}, {op: opSelect, recipe: a},
		})
	})
	// The case the superseded state was first admitted for: runs of two
	// objectives alternate on one model, each under its own recipe.
	t.Run("objectives alternating on one model", func(t *testing.T) {
		runLifecycle(t, []lifecycleStep{
			{op: opSelect, recipe: a}, {op: opSelect, recipe: b}, {op: opSelect, recipe: a}, {op: opSelect, recipe: b},
		})
	})
	// A slot may come back to one recipe more than once, and from a recipe
	// that was verified again after it took the slot.
	t.Run("rolled back to the same recipe twice", func(t *testing.T) {
		const c = 2
		runLifecycle(t, []lifecycleStep{
			{op: opSelect, recipe: a}, {op: opSelect, recipe: b}, {op: opSelect, recipe: b}, {op: opRetire}, {op: opRollback},
			{op: opSelect, recipe: c}, {op: opRetire}, {op: opRollback}, {op: opSelect, recipe: a},
		})
	})
	for sequence := range generatedLifecycles {
		steps := generateLifecycle(rand.New(rand.NewPCG(uint64(sequence), lifecycleSeed)))
		t.Run(fmt.Sprintf("generated %d", sequence), func(t *testing.T) { runLifecycle(t, steps) })
	}
}
