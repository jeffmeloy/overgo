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
	// opRelease releases the slot a retired recipe still holds.
	opRelease = "release"
	// opRestart closes the store and opens it.
	opRestart = "restart"
	// opReplay sends the last lifecycle message again.
	opReplay = "replay"
	// Three messages a lifecycle refuses: an activation no decision accepted,
	// an activation on proof made over another recipe, and a retirement on
	// proof that passed.
	opUnproved  = "activate-undecided"
	opMisproved = "activate-on-another's-proof"
	opUnfailed  = "retire-on-passed-proof"
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

// The failures a reference can be made to have. A store is held to the
// reference, so a reference with a failure differs from a sound store exactly
// where a store with that failure would differ from the sound reference: a
// sequence that tells them apart is one that catches the failure.
const (
	sound                 = ""
	faultSelectionRestore = "selecting a recipe makes the one retired before it trainable again"
	faultNoSupersession   = "an activation leaves the recipe it took the slot from active"
	faultSlotStays        = "an activation leaves the slot naming the recipe that held it"
	faultUnproved         = "an activation no decision accepted is recorded"
	faultMisproved        = "an activation on proof made over another recipe is accepted"
	faultUnfailed         = "a retirement on proof that passed retires the recipe"
	faultNoRetirement     = "a retirement on failed proof leaves the recipe as it was"
	faultRetirementDrops  = "a retirement takes the slot from the recipe it retires"
	faultReleaseKeeps     = "a release leaves the slot naming the retired recipe"
	faultReleaseAlways    = "a slot is released whatever it names, and when it names nothing"
)

// lifecycleStep is one thing that happens, and the recipe it happens to.
type lifecycleStep struct {
	op     string
	recipe int
}

func (s lifecycleStep) String() string {
	switch s.op {
	case opRetire, opRollback, opRelease, opRestart, opReplay, opUnfailed:
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
// used, unless it is given the failure that does.
type lifecycleModel struct {
	recipes  []recipeRecord
	selected int
	fault    string
}

func newLifecycleModel(fault string) *lifecycleModel {
	model := &lifecycleModel{recipes: make([]recipeRecord, lifecycleRecipes), selected: noRecipe, fault: fault}
	for index := range model.recipes {
		model.recipes[index] = recipeRecord{status: recipe.StatusCandidate, over: noRecipe}
	}
	return model
}

// active returns the selected recipe when it is active, and noRecipe when the
// slot names a retired recipe or none.
func (m *lifecycleModel) active() int {
	if m.selected != noRecipe && m.recipes[m.selected].status == recipe.StatusActive {
		return m.selected
	}
	return noRecipe
}

// refuses reports whether a run asking to train under the recipe is refused.
func (m *lifecycleModel) refuses(index int) bool {
	if m.fault == faultSelectionRestore && m.active() != noRecipe && m.selected != index {
		return false
	}
	return m.recipes[index].refused
}

// activates reports whether an activation of the recipe would take the slot:
// the recipe is a candidate, validated or not.
func (m *lifecycleModel) activates(index int) bool {
	status := m.recipes[index].status
	return status == recipe.StatusCandidate || status == recipe.StatusValidated
}

// predecessor returns the recipe a rollback restores: the one the selected
// recipe's activation took the slot from, when the selected recipe is retired.
func (m *lifecycleModel) predecessor() int {
	if m.selected == noRecipe || !m.recipes[m.selected].refused {
		return noRecipe
	}
	return m.recipes[m.selected].over
}

// reinstates reports whether the recipe is one a reinstatement activates: a
// superseded recipe that does not hold the slot.
func (m *lifecycleModel) reinstates(index int) bool {
	return m.recipes[index].status == recipe.StatusSuperseded && m.selected != index
}

// releases reports whether a release finds a slot to release: one that names
// a retired recipe.
func (m *lifecycleModel) releases() bool {
	return m.fault == faultReleaseAlways || m.selected != noRecipe && m.recipes[m.selected].refused
}

// take gives a recipe the slot, active. The recipe that held it loses the
// slot and nothing else: it is superseded and stays as eligible as it was, and
// one already retired keeps its refusal.
func (m *lifecycleModel) take(index int) {
	held := m.selected
	if held != noRecipe && held != index && m.recipes[held].status == recipe.StatusActive && m.fault != faultNoSupersession {
		m.recipes[held].status = recipe.StatusSuperseded
	}
	if m.recipes[index] = (recipeRecord{status: recipe.StatusActive, over: held}); held == noRecipe || m.fault != faultSlotStays {
		m.selected = index
	}
}

// retire retires the active recipe on failed proof: it is superseded and
// refused, and keeps the slot.
func (m *lifecycleModel) retire() int {
	retired := m.selected
	m.recipes[retired].status, m.recipes[retired].refused = recipe.StatusSuperseded, true
	if m.fault == faultRetirementDrops {
		m.selected = noRecipe
	}
	return retired
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
		if !m.recipes[step.recipe].refused && m.activates(step.recipe) {
			m.take(step.recipe)
		}
	case opUnproved:
		if m.fault == faultUnproved && m.activates(step.recipe) {
			m.take(step.recipe)
		}
	case opMisproved:
		// The activation validates a candidate before it reads the proof, so a
		// candidate it refuses is left validated.
		if m.fault == faultMisproved && m.activates(step.recipe) {
			m.take(step.recipe)
		} else if m.recipes[step.recipe].status == recipe.StatusCandidate {
			m.recipes[step.recipe].status = recipe.StatusValidated
		}
	case opRetire:
		if m.active() != noRecipe && m.fault != faultNoRetirement {
			return m.retire()
		}
	case opUnfailed:
		if m.fault == faultUnfailed && m.active() != noRecipe {
			return m.retire()
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
	case opRelease:
		if m.releases() && m.fault != faultReleaseKeeps {
			m.selected = noRecipe
		}
	}
	return noRecipe
}

// lifecycleReading is what a store says of a model's training recipes: each
// one's state, the refusal a run asking to train under it meets, which recipe
// the slot names, whether the model declares an active recipe, and which
// recipe its verified activation is.
type lifecycleReading struct {
	statuses []recipe.Status
	refusals []error
	selected int
	declared bool
	active   int
}

// differs says where a reading departs from the reference, or nothing.
func (m *lifecycleModel) differs(reading lifecycleReading) string {
	if reading.selected != m.selected {
		return fmt.Sprintf("the store's slot names recipe %s and the reference's %s", recipeName(reading.selected), recipeName(m.selected))
	}
	if active := m.active(); reading.active != active || reading.declared != (active != noRecipe) {
		return fmt.Sprintf("the store's verified activation is recipe %s, declared %t, and the reference's is %s", recipeName(reading.active), reading.declared, recipeName(active))
	}
	for index, record := range m.recipes {
		if reading.statuses[index] != record.status {
			return fmt.Sprintf("recipe %s is %q in the store and %q in the reference", recipeName(index), reading.statuses[index], record.status)
		}
		if refused := reading.refusals[index] != nil; refused != m.refuses(index) {
			return fmt.Sprintf("recipe %s: a run asking to train under it is refused %t (%v), and the reference refuses it %t",
				recipeName(index), refused, reading.refusals[index], m.refuses(index))
		}
	}
	return ""
}

// recipeLifecycle drives one model's training recipes in a store through the
// doors a run, a retirement, a reinstatement, a rollback and a release use.
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

func (f *recipeLifecycle) close() {
	f.t.Helper()
	if err := f.store.Close(); err != nil {
		f.t.Fatal(err)
	}
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
func (f *recipeLifecycle) proof(index int, outcome runrecord.Outcome) modelrecipe.Verification {
	f.t.Helper()
	f.proofs++
	step, failure := runrecord.StepSucceeded, ""
	if outcome == runrecord.OutcomeFailed {
		step, failure = runrecord.StepFailed, "exact-mismatch"
	}
	record, err := runrecord.NewGateRecord(f.recipes[index].ID, f.environment, strings.Repeat("f0", 20), outcome, failure, f.proofs, []runrecord.GateStep{{
		Name: "verify", Phase: runrecord.PhaseValidate, Outcome: step, DurationNS: f.proofs,
	}})
	if err != nil {
		f.t.Fatal(err)
	}
	f.commit(record.Batch(fmt.Sprintf("eligibility/proof/%d", f.proofs)))
	return modelrecipe.Verification{Gate: record.Result.ID, Run: record.Run.ID}
}

// index returns which of the model's recipes id is.
func (f *recipeLifecycle) index(id artifact.ID) int {
	f.t.Helper()
	for index, definition := range f.recipes {
		if definition.ID == id {
			return index
		}
	}
	f.t.Fatalf("%s is none of the model's recipes", id)
	return noRecipe
}

// selected returns the recipe the model's training slot names.
func (f *recipeLifecycle) selected() (int, *artifact.ID) {
	f.t.Helper()
	id, bound, err := artifact.ResolveAlias(f.t.Context(), f.store, modelrecipe.ActiveAliasPrefix(recipe.TaskTraining)+f.model.String())
	if err != nil {
		f.t.Fatal(err)
	}
	if !bound {
		return noRecipe, nil
	}
	return f.index(id), &id
}

// retire retires the selected recipe on proof with the outcome given: proof
// that failed retires it, and proof that passed is refused. The proof is over
// the selected recipe or, every other time, over the recipe after it, as a
// candidate meant to replace it: either is proof about the model's training.
func (f *recipeLifecycle) retire(outcome runrecord.Outcome) error {
	f.t.Helper()
	index, _ := f.selected()
	if index == noRecipe {
		return errors.New("no recipe is selected")
	}
	candidate := (index + int(f.proofs%uint64(len(f.recipes)-1))) % len(f.recipes)
	ctx, proved := f.t.Context(), f.proof(candidate, outcome)
	f.last = func() error {
		return modelrecipe.RetireActiveCapability(ctx, f.store, f.recipes[candidate], proved, "its proof was run again")
	}
	return f.last()
}

// reinstate activates a superseded recipe again on fresh proof, over whichever
// recipe holds the slot.
func (f *recipeLifecycle) reinstate(index int) error {
	f.t.Helper()
	ctx, fresh := f.t.Context(), f.proof(index, runrecord.OutcomeSucceeded)
	_, held := f.selected()
	key := fmt.Sprintf("eligibility/reinstated/%d", f.proofs)
	f.last = func() error {
		_, _, err := modelrecipe.ActivateVerified(ctx, f.store, key, f.recipes[index], fresh, recipe.EvidenceVerified, "reinstated on fresh proof", nil, held)
		return err
	}
	return f.last()
}

// rollback returns the slot to the recipe the retired holder's activation took
// it from, on fresh proof over the recipe named.
func (f *recipeLifecycle) rollback(restored int) error {
	f.t.Helper()
	ctx, fresh := f.t.Context(), f.proof(restored, runrecord.OutcomeSucceeded)
	f.last = func() error {
		return modelrecipe.RollbackActivation(ctx, f.store, f.model, recipe.TaskTraining, fresh, recipe.EvidenceVerified, "restored on fresh proof")
	}
	return f.last()
}

// unproved records an activation of the recipe with a verifier's gate and run
// and no decision that accepted them.
func (f *recipeLifecycle) unproved(index int) error {
	f.t.Helper()
	proved := f.proof(index, runrecord.OutcomeSucceeded)
	_, _, err := modelrecipe.Transition(f.t.Context(), f.store, fmt.Sprintf("eligibility/unproved/%d", f.proofs), f.recipes[index],
		recipe.StatusActive, []artifact.ID{proved.Gate, proved.Run}, nil)
	return err
}

// misproved activates the recipe on proof made over the recipe after it.
func (f *recipeLifecycle) misproved(index int) error {
	f.t.Helper()
	another := f.proof((index+1)%len(f.recipes), runrecord.OutcomeSucceeded)
	return modelrecipe.ActivateCapability(f.t.Context(), f.store, f.recipes[index], another, recipe.EvidenceVerified, "proof made over another recipe")
}

func (f *recipeLifecycle) read() lifecycleReading {
	f.t.Helper()
	ctx := f.t.Context()
	reading := lifecycleReading{statuses: make([]recipe.Status, len(f.recipes)), refusals: make([]error, len(f.recipes)), active: noRecipe}
	reading.selected, _ = f.selected()
	for index, definition := range f.recipes {
		status, _, err := modelrecipe.Status(ctx, f.store, definition.ID)
		if err != nil {
			f.t.Fatal(err)
		}
		reading.statuses[index] = status
		_, reading.refusals[index] = ResolveSessionAuthority(ctx, f.store, f.model, definition.ID)
	}
	declared, err := modelrecipe.HasActiveRecipe(ctx, f.store, f.model, recipe.TaskTraining)
	if err != nil {
		f.t.Fatal(err)
	}
	reading.declared = declared
	// A slot that names a retired recipe is no activation: its record is an error.
	if activation, found, err := modelrecipe.ActiveRecord(ctx, f.store, f.model, recipe.TaskTraining); err == nil && found {
		reading.active = f.index(activation.Definition.ID)
	}
	return reading
}

// perform sends one step through the store's doors and says how the door's
// answer departs from what the reference, before the step, expects of it.
func (f *recipeLifecycle) perform(step lifecycleStep, model *lifecycleModel) string {
	f.t.Helper()
	answered := func(err error, accepted bool) string {
		if (err == nil) != accepted {
			return fmt.Sprintf("the store answered %v, and the reference accepts the step: %t", err, accepted)
		}
		return ""
	}
	switch step.op {
	case opSelect, opRetry:
		f.session(step.recipe, nil)
	case opFail:
		f.session(step.recipe, errors.New("the run failed"))
	case opCancel:
		f.session(step.recipe, context.Canceled)
	case opRetire:
		return answered(f.retire(runrecord.OutcomeFailed), model.active() != noRecipe)
	case opUnfailed:
		return answered(f.retire(runrecord.OutcomeSucceeded), false)
	case opUnproved:
		return answered(f.unproved(step.recipe), false)
	case opMisproved:
		return answered(f.misproved(step.recipe), false)
	case opReinstate:
		if model.reinstates(step.recipe) {
			return answered(f.reinstate(step.recipe), true)
		}
	case opRollback:
		// A rollback with nothing to restore is refused whatever proof it brings.
		restored := model.predecessor()
		return answered(f.rollback(max(restored, 0)), restored != noRecipe)
	case opRelease:
		return answered(modelrecipe.ReleaseRetiredAlias(f.t.Context(), f.store, f.model, recipe.TaskTraining), model.releases())
	case opRestart:
		f.close()
		f.open()
	case opReplay:
		// A message sent again may be accepted or refused; what it may not do
		// is change anything, which the reading after it holds.
		if f.last != nil {
			f.t.Logf("%s: the message sent again answered %v", step, f.last())
		}
	}
	return ""
}

// lifecycleDifference takes one model's recipes through steps in a store and
// holds the store, after each step, to two things: no step changed whether a
// recipe may be trained under but for the one recipe the step is about, and
// the store reads as the reference does. It returns the first departure, or
// nothing.
func lifecycleDifference(t *testing.T, model *lifecycleModel, steps []lifecycleStep) string {
	t.Helper()
	lifecycle := newRecipeLifecycle(t)
	defer lifecycle.close()
	before := lifecycle.read()
	if difference := model.differs(before); difference != "" {
		return "before any step: " + difference
	}
	for index, step := range steps {
		difference := lifecycle.perform(step, model)
		subject := model.apply(step)
		after := lifecycle.read()
		for other := range after.refusals {
			if difference == "" && other != subject && (after.refusals[other] == nil) != (before.refusals[other] == nil) {
				difference = fmt.Sprintf("the step changed whether a run may train under recipe %s, which it is not about: refused %v before and %v after",
					recipeName(other), before.refusals[other], after.refusals[other])
			}
		}
		if difference == "" {
			difference = model.differs(after)
		}
		if difference != "" {
			return fmt.Sprintf("after %v: %s", steps[:index+1], difference)
		}
		before = after
	}
	return ""
}

// lifecycleSequence is a named sequence of steps.
type lifecycleSequence struct {
	name  string
	steps []lifecycleStep
}

// lifecycleSequences are the sequences a store is taken through: four that
// are fixed, and the generated ones. A generated sequence draws its steps; a
// retry is of the recipe the run before it trained under.
func lifecycleSequences() []lifecycleSequence {
	const a, b, c = 0, 1, 2
	sequences := []lifecycleSequence{
		// The sequence the rule was written for: A is retired, B is selected,
		// and A is refused until it is reinstated.
		{"retired, another selected, asked for", []lifecycleStep{
			{opSelect, a}, {opRetire, a}, {opSelect, b}, {opFail, a}, {opReinstate, a}, {opSelect, a},
		}},
		// The case the superseded state was first admitted for: runs of two
		// objectives alternate on one model, each under its own recipe.
		{"objectives alternating on one model", []lifecycleStep{{opSelect, a}, {opSelect, b}, {opSelect, a}, {opSelect, b}}},
		// A slot may come back to one recipe more than once, and from a recipe
		// that was verified again after it took the slot.
		{"rolled back to the same recipe twice", []lifecycleStep{
			{opSelect, a}, {opSelect, b}, {opSelect, b}, {opRetire, b}, {opRollback, a}, {opSelect, c}, {opRetire, c}, {opRollback, a}, {opSelect, a},
		}},
		// The messages a lifecycle refuses change nothing, a refused activation
		// leaves a candidate validated and still activatable, and a slot is
		// released once, when the recipe it names is retired.
		{"refused messages, and a slot released", []lifecycleStep{
			{opSelect, a}, {opUnproved, b}, {opMisproved, b}, {opSelect, b}, {opUnfailed, b}, {opRelease, b},
			{opRetire, b}, {opRelease, b}, {opRelease, b}, {opSelect, c}, {opRelease, c},
		}},
	}
	ops := []string{
		opSelect, opSelect, opSelect, opSelect, opRetry, opFail, opCancel, opRetire, opRetire, opRetire, opReinstate, opReinstate,
		opRollback, opRollback, opRelease, opUnproved, opMisproved, opUnfailed, opRestart, opReplay,
	}
	for sequence := range generatedLifecycles {
		random := rand.New(rand.NewPCG(uint64(sequence), lifecycleSeed))
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
		sequences = append(sequences, lifecycleSequence{fmt.Sprintf("generated %d", sequence), steps})
	}
	return sequences
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
// Four sequences are fixed and the rest are generated: runs that succeed, fail
// and are cancelled, retries, retirements, reinstatements, rollbacks, releases
// of a retired recipe's slot, messages a lifecycle refuses, restarts of the
// store and messages sent twice, over three recipes of one model. Each is run
// against a real store through the doors a training run, a retirement, a
// reinstatement, a rollback and a release use, and after every step the store
// is held to a reference that keeps eligibility and selection apart.
func TestSelectionNeverRestoresEligibility(t *testing.T) {
	stubExecutableCodeCommit(t, strings.Repeat("e1", 20))
	for _, sequence := range lifecycleSequences() {
		t.Run(sequence.name, func(t *testing.T) {
			if difference := lifecycleDifference(t, newLifecycleModel(sound), sequence.steps); difference != "" {
				t.Fatal(difference)
			}
		})
	}
}

// TestEveryRetiredLifecycleFailureIsStillCaught holds the sequences to the
// failures the tests they replaced were written to catch. Three tests of
// internal/modelrecipe were retired for the sequences: each built its own
// store and recipes to walk one path through the lifecycle and assert the
// states along it. For every failure one of them caught, the reference is
// given that failure, and some sequence must tell it from the store; a
// failure no sequence tells apart is one the retirement lost. The first
// failure is the one this rule was written against, which no test caught.
func TestEveryRetiredLifecycleFailureIsStillCaught(t *testing.T) {
	stubExecutableCodeCommit(t, strings.Repeat("e1", 20))
	sequences := lifecycleSequences()
	for _, failure := range []struct{ fault, caughtBy string }{
		{faultSelectionRestore, "no test: SQA-099"},
		{faultNoSupersession, "TestLifecyclePromotionAndSupersession: the first recipe is superseded when the second is activated"},
		{faultSlotStays, "TestLifecyclePromotionAndSupersession: the active record is the second recipe"},
		{faultUnproved, "TestLifecyclePromotionAndSupersession: an ordinary transition cannot bypass verified activation"},
		{faultMisproved, "TestLifecyclePromotionAndSupersession: an activation refuses verifier output for another recipe"},
		{faultUnfailed, "TestRetireActiveCapabilityRequiresFailedEvidence: successful evidence retires nothing"},
		{faultNoRetirement, "TestRetireActiveCapabilityRequiresFailedEvidence: the retired recipe is superseded and no longer declared"},
		{faultRetirementDrops, "TestRetireActiveCapabilityRequiresFailedEvidence: a retirement keeps the slot"},
		{faultReleaseKeeps, "TestRetireActiveCapabilityRequiresFailedEvidence: a release removes the retired recipe's slot"},
		{faultReleaseAlways, "TestRetireActiveCapabilityRequiresFailedEvidence: a slot is released once; TestReleaseRetiredAliasRefusesActiveRecipe: an active slot stays bound"},
	} {
		caught := ""
		for _, sequence := range sequences {
			if caught = lifecycleDifference(t, newLifecycleModel(failure.fault), sequence.steps); caught != "" {
				t.Logf("%s: %s: %s", failure.fault, sequence.name, caught)
				break
			}
		}
		if caught == "" {
			t.Errorf("no sequence fails when %s, the failure once caught by %s", failure.fault, failure.caughtBy)
		}
	}
}
