package evaluation

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/runrecord"
	"overgo/internal/tensorstats"
	"overgo/internal/testutil"
)

// exactInstructionPower is instructionPower in rational arithmetic, from the
// binomial coefficients: over every count of discordant pairs and of wins
// among them, the chance of the counts the exact test passes at level.
func exactInstructionPower(groups int64, discordance, win, level *big.Rat) *big.Rat {
	term := func(trials, successes int64, chance *big.Rat) *big.Rat {
		value := new(big.Rat).SetInt(new(big.Int).Binomial(trials, successes))
		for range successes {
			value.Mul(value, chance)
		}
		for range trials - successes {
			value.Mul(value, new(big.Rat).Sub(big.NewRat(1, 1), chance))
		}
		return value
	}
	power := new(big.Rat)
	for discordant := int64(0); discordant <= groups; discordant++ {
		for wins := int64(0); wins <= discordant; wins++ {
			tail := new(big.Rat)
			for more := wins; more <= discordant; more++ {
				tail.Add(tail, term(discordant, more, big.NewRat(1, 2)))
			}
			if tail.Cmp(level) <= 0 {
				power.Add(power, new(big.Rat).Mul(term(groups, discordant, discordance), term(discordant, wins, win)))
			}
		}
	}
	return power
}

// TestInstructionAcceptanceBindsProtocolGroupsAndComparisons holds the
// instruction pilot's acceptance to what it fixes before any arm is trained,
// and a confirmation under it to that.
//
// Units. Prompts that ask for one combination of rules are one group, a rule
// asked twice counted once; one prompt stands for each, the same whatever
// order the cases are listed in and another under another seed; every group
// is wholly in one part; and the promotion part's groups are the acceptance's
// count of independent pairs; and the prompts of each part are named before
// any reading, the same the acceptance then holds. Its power is the exact
// chance of a gain's being shown over those groups, held to an enumeration in
// rational arithmetic, where arms differ on at most the upper bound of the
// share of development groups the base differs on read twice, and the gain
// more. An acceptance whose groups cannot show the gain is refused as not
// enough evidence, and for that reason alone; one that leaves a gain, a
// development reading, a comparison, a protocol setting or the training
// specification unnamed is refused otherwise.
//
// A confirmation. It is refused, with nothing charged and nothing read, for a
// comparison the acceptance did not name, for answers made under another
// output limit, sampling, window or scorer than it fixed, for an acceptance
// whose count of groups is not its promotion part's, and for a split that has
// one group in two parts. A named comparison is charged before its prompts
// are read, read once, and its verdict is the exact test of the pairs the
// arms differ on: prompts that stand for no group enter nothing, so an arm
// far better over every prompt and no better on the groups' own is not shown
// better. The comparison asked again returns its verdict unread; asked for
// another candidate it is refused.
//
// A replacement acceptance over the same prompts starts with what they have
// answered spent, and so does one over other prompts of the same groups: the
// group is what answered, whichever prompt stood for it. A likelihood pool's acceptance is still read as what it is:
// neither kind confirms under the other's rule.
func TestInstructionAcceptanceBindsProtocolGroupsAndComparisons(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, objective, training, heldout := poolObjective(t)
	const combinations = 140
	var cases []InstructionCase
	for index, record := range slices.Concat(training.Records, heldout.Records) {
		rules := []string{fmt.Sprintf("length:%d", index%combinations), fmt.Sprintf("keywords:%d", index%combinations)}
		if index >= combinations {
			// The same combination, one rule asked a second time.
			rules = append(rules, rules[0]+"/2")
		}
		cases = append(cases, InstructionCase{Record: record.ID, Rules: rules})
	}
	model := func(name string) artifact.ID {
		return testutil.ArtifactID(t, artifact.KindModel, "instruction fixture "+name)
	}
	protocol := InstructionProtocol{Scorer: InstructionScorer, Template: "chat-template", Thinking: "off", Sampling: "greedy", Stop: "end of turn", Window: 4096, MaxTokens: 1280}
	request := InstructionAcceptanceRequest{
		Dataset: objective.Dataset, Base: model("base"), Training: testutil.ArtifactID(t, artifact.KindProfile, "instruction fixture training"),
		Cases: cases, Protocol: protocol, Seed: 7, Weights: [poolPartCount]uint64{1, 1, 4, 1}, Gain: 0.2,
		Comparisons: []InstructionComparison{{Name: "reward over base", Candidate: "reward", Baseline: "base"}, {Name: "reward over control", Candidate: "reward", Baseline: "control"}},
		Pairs:       200, Discordant: 10, Basis: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvaluation, "instruction fixture development reading")},
	}
	// The training specification is committed before an acceptance names it.
	if _, err := artifact.Publish(ctx, store, artifact.Batch{Key: "instruction-fixture/training", Artifacts: []artifact.Descriptor{{ID: request.Training}}}); err != nil {
		t.Fatal(err)
	}
	publish := func(request InstructionAcceptanceRequest) (InstructionAcceptance, runrecord.SplitPartition) {
		t.Helper()
		acceptance, batch, err := BuildInstructionAcceptance(request)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.Publish(ctx, store, batch); err != nil {
			t.Fatal(err)
		}
		content, _, err := artifact.ReadContent(ctx, store, acceptance.Split)
		if err != nil {
			t.Fatal(err)
		}
		partition, err := runrecord.ParseSplitPartition(content.Data)
		if err != nil {
			t.Fatal(err)
		}
		return acceptance, partition
	}
	acceptance, partition := publish(request)

	// Units: one prompt a group, a group in one part, the promotion part's groups the count.
	parts := map[string]string{}
	var promotion []string
	var promoted, developed dataset.Record
	for at, shard := range []artifact.ID{partition.Development, partition.Selection, partition.Promotion, partition.Audit} {
		membership, found, err := dataset.LoadMembership(ctx, store, shard)
		if err != nil || !found {
			t.Fatalf("part %s is committed %t (%v)", PoolParts[at], found, err)
		}
		for _, record := range membership.Records {
			if held, found := parts[record.Group]; found {
				t.Fatalf("group %s is in %s and %s", record.Group, held, PoolParts[at])
			}
			switch parts[record.Group] = PoolParts[at]; at {
			case poolPromotion:
				promotion, promoted = append(promotion, record.ID), record
			case poolDevelopment:
				developed = record
			}
		}
	}
	if len(parts) != combinations || uint64(len(promotion)) != acceptance.Groups {
		t.Fatalf("%d cases of %d rule combinations are %d groups, %d of them the promotion part's and %d the acceptance's count", len(cases), combinations, len(parts), len(promotion), acceptance.Groups)
	}
	reversed := request
	reversed.Cases = slices.Clone(cases)
	slices.Reverse(reversed.Cases)
	if again, _, err := BuildInstructionAcceptance(reversed); err != nil || again.ID != acceptance.ID {
		t.Fatalf("the same cases listed in another order fix acceptance %s (%v), want %s", again.ID, err, acceptance.ID)
	}
	reseeded := request
	reseeded.Seed = 8
	if other, _, err := BuildInstructionAcceptance(reseeded); err != nil || other.Split == acceptance.Split {
		t.Fatalf("another seed draws split %s (%v), want another than %s", other.Split, err, acceptance.Split)
	}

	// Power: the acceptance's own, and the function against an enumeration.
	_, repeated := tensorstats.ExactShareBounds(request.Discordant, request.Pairs, falseGain)
	if acceptance.Power < 1-missedGain || repeated <= float64(request.Discordant)/float64(request.Pairs) || acceptance.Discordance != repeated+request.Gain || acceptance.FalseGain != falseGain {
		t.Fatalf("the acceptance states power %.3f at %.3f of pairs discordant, want at least %.2f at the gain more than the upper bound %.3f of the %d of %d groups seen to differ read twice", acceptance.Power, acceptance.Discordance, 1-missedGain, repeated, request.Discordant, request.Pairs)
	}
	named, err := InstructionGroups(request)
	if err != nil || !slices.Equal(named[poolPromotion], slices.Sorted(slices.Values(promotion))) || !slices.Contains(named[poolDevelopment], developed.ID) {
		t.Fatalf("the groups named before the acceptance is fixed = (%d promotion prompts, %v), want the %d its promotion part holds", len(named[poolPromotion]), err, len(promotion))
	}
	want, _ := exactInstructionPower(9, big.NewRat(1, 2), big.NewRat(4, 5), big.NewRat(1, 20)).Float64()
	if got := instructionPower(9, 0.5, 0.3, 0.05); math.Abs(got-want) > 1e-12 {
		t.Fatalf("the power of 9 groups = %v, want %v", got, want)
	}
	for name, changed := range map[string]func(*InstructionAcceptanceRequest){
		"too few groups":         func(r *InstructionAcceptanceRequest) { r.Weights = [poolPartCount]uint64{6, 6, 1, 6} },
		"no gain":                func(r *InstructionAcceptanceRequest) { r.Gain = 0 },
		"no development reading": func(r *InstructionAcceptanceRequest) { r.Pairs, r.Discordant = 0, 0 },
		"no comparison":          func(r *InstructionAcceptanceRequest) { r.Comparisons = nil },
		"no output limit":        func(r *InstructionAcceptanceRequest) { r.Protocol.MaxTokens = 0 },
		"no training":            func(r *InstructionAcceptanceRequest) { r.Training = artifact.ID{} },
	} {
		refused := request
		changed(&refused)
		_, _, err := BuildInstructionAcceptance(refused)
		if err == nil || (name == "too few groups") != errors.Is(err, ErrInstructionEvidence) {
			t.Errorf("an acceptance with %s = %v, want it refused", name, err)
		}
	}

	// A confirmation.
	charged := func(acceptance InstructionAcceptance, partition runrecord.SplitPartition) int {
		t.Helper()
		granted, err := runrecord.NewBudget(poolUnit, partition.Promotion, uint64(len(acceptance.Comparisons)), acceptance.ID)
		if err != nil {
			t.Fatal(err)
		}
		_, charges, err := runrecord.BudgetCharges(ctx, store, granted.ID)
		if err != nil {
			t.Fatal(err)
		}
		return len(charges)
	}
	readings := 0
	// reading answers for the promotion part's prompts: the candidate passes wins of them the baseline fails, and both pass the rest.
	reading := func(wins int) func([]string) ([]bool, []bool, error) {
		return func(prompts []string) ([]bool, []bool, error) {
			readings++
			if !slices.Equal(prompts, promotion) || charged(acceptance, partition) == 0 {
				t.Errorf("a reading was handed %d prompts with %d charges committed, want the promotion part's %d after its charge", len(prompts), charged(acceptance, partition), len(promotion))
			}
			passed, base := make([]bool, len(prompts)), make([]bool, len(prompts))
			for index := range prompts {
				passed[index], base[index] = true, index >= wins
			}
			return passed, base, nil
		}
	}
	reward, control := model("reward"), model("control")
	limited, sampled, windowed, rescored := protocol, protocol, protocol, protocol
	limited.MaxTokens, sampled.Sampling, windowed.Window, rescored.Scorer = 512, "temperature 0.7", 2048, "instruction-rules/strict/0"
	miscounted := acceptance
	miscounted.Groups, miscounted.ID = acceptance.Groups+1, artifact.ID{}
	// A development part from another split, which holds a prompt of the promotion part.
	stray, err := dataset.SplitOf(objective.Dataset, request.Seed, []dataset.PartitionRecords{
		{Name: "development", Records: []dataset.Record{promoted}}, {Name: "rest", Records: []dataset.Record{developed}},
	})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := stray.PublicationBatch("instruction-fixture/stray", nil)
	if err != nil {
		t.Fatal(err)
	}
	development := stray.Memberships[slices.IndexFunc(stray.Memberships, func(membership dataset.Membership) bool { return membership.Partition == "development" })]
	leaking, err := runrecord.NewSplitPartition(objective.Dataset, development.ID, partition.Selection, partition.Promotion, partition.Audit, partition.Proposer)
	if err != nil {
		t.Fatal(err)
	}
	leaked := acceptance
	leaked.Split, leaked.ID = leaking.ID, artifact.ID{}
	part, err := leaking.Batch("instruction-fixture/leaking")
	if err != nil {
		t.Fatal(err)
	}
	batch.Contents, batch.Lineage = append(batch.Contents, part.Contents...), append(batch.Lineage, part.Lineage...)
	for name, tampered := range map[string]*InstructionAcceptance{"miscounted": &miscounted, "leaked": &leaked} {
		if *tampered, err = instructionAcceptanceCodec.NewInitial(*tampered); err != nil {
			t.Fatal(err)
		}
		part, err := instructionAcceptanceCodec.Batch("instruction-fixture/"+name, *tampered, artifact.DependencyLineage(tampered.ID, tampered.Split), nil)
		if err != nil {
			t.Fatal(err)
		}
		// Each is published with the grant an acceptance has, so that it is its own fault it is refused for.
		granted, err := runrecord.NewBudget(poolUnit, partition.Promotion, uint64(len(tampered.Comparisons)), tampered.ID)
		if err != nil {
			t.Fatal(err)
		}
		grant, err := granted.Batch("instruction-fixture/" + name)
		if err != nil {
			t.Fatal(err)
		}
		batch.Contents, batch.Lineage = append(append(batch.Contents, part.Contents...), grant.Contents...), append(append(batch.Lineage, part.Lineage...), grant.Lineage...)
	}
	if _, err := artifact.Publish(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	for name, asked := range map[string]struct {
		acceptance artifact.ID
		protocol   InstructionProtocol
		comparison string
		want       string
	}{
		"a comparison the acceptance did not name": {acceptance.ID, protocol, "control over base", "not one of them"},
		"another output limit":                     {acceptance.ID, limited, "reward over base", "nothing was charged"},
		"another sampling":                         {acceptance.ID, sampled, "reward over base", "nothing was charged"},
		"another window":                           {acceptance.ID, windowed, "reward over base", "nothing was charged"},
		"another scorer":                           {acceptance.ID, rescored, "reward over base", "nothing was charged"},
		"a count of groups that is not the part's": {miscounted.ID, protocol, "reward over base", "was fixed for"},
		"a split with one group in two parts":      {leaked.ID, protocol, "reward over base", "wholly in one"},
		"a comparison with no name":                {acceptance.ID, protocol, "", "not one of them"},
	} {
		if _, _, err := confirmInstruction(ctx, store, asked.acceptance, asked.protocol, asked.comparison, reward, request.Base, reading(len(promotion))); err == nil || !strings.Contains(err.Error(), asked.want) || readings != 0 || charged(acceptance, partition) != 0 {
			t.Fatalf("%s = %v with %d readings and %d charges, want it refused with nothing read or charged", name, err, readings, charged(acceptance, partition))
		}
	}

	const wins = 30
	gain, left, err := confirmInstruction(ctx, store, acceptance.ID, protocol, "reward over base", reward, request.Base, reading(wins))
	if err != nil || gain.Verdict != poolGain || left != 1 || readings != 1 || gain.Candidate != reward || gain.Acceptance != acceptance.ID {
		t.Fatalf("a candidate that passes %d groups the baseline fails and loses none = (%+v, %d left, %v) after %d readings, want a gain, one comparison left and one reading", wins, gain, left, err, readings)
	}
	if recorded, left, err := confirmInstruction(ctx, store, acceptance.ID, protocol, "reward over base", reward, request.Base, reading(0)); err != nil || recorded.ID != gain.ID || left != 1 || readings != 1 {
		t.Fatalf("the comparison asked again = (%s, %d left, %v) after %d readings, want its recorded verdict %s unread", recorded.ID, left, err, readings, gain.ID)
	}
	if _, _, err := confirmInstruction(ctx, store, acceptance.ID, protocol, "reward over base", model("another checkpoint"), request.Base, reading(wins)); err == nil || !strings.Contains(err.Error(), "another candidate") || readings != 1 {
		t.Fatalf("the comparison asked for another candidate = %v after %d readings, want it refused", err, readings)
	}
	// Over every prompt the arm would be far better: each group's other prompts
	// are no part of the reading, and on the groups' own it wins three and loses one.
	few := func(prompts []string) ([]bool, []bool, error) {
		readings++
		passed, base := make([]bool, len(prompts)), make([]bool, len(prompts))
		for index := range prompts {
			passed[index], base[index] = index != 3, index >= 3
		}
		return passed, base, nil
	}
	undecided, left, err := confirmInstruction(ctx, store, acceptance.ID, protocol, "reward over control", reward, control, few)
	if err != nil || undecided.Verdict != poolUndecided || left != 0 || readings != 2 {
		t.Fatalf("three wins and a loss among %d groups = (%q, %d left, %v), want not enough evidence and nothing left", len(promotion), undecided.Verdict, left, err)
	}

	// A replacement over the same prompts, with a third comparison named.
	replaced := request
	replaced.Comparisons = append(slices.Clone(request.Comparisons), InstructionComparison{Name: "control over base", Candidate: "control", Baseline: "base"})
	second, same := publish(replaced)
	if same.Promotion != partition.Promotion || second.ID == acceptance.ID {
		t.Fatalf("the replacement is acceptance %s over part %s, want another acceptance over the first's part", second.ID, same.Promotion)
	}
	if _, left, err := confirmInstruction(ctx, store, second.ID, protocol, "control over base", control, request.Base, reading(wins)); err != nil || left != 0 || readings != 3 {
		t.Fatalf("the replacement's first comparison leaves %d (%v), want none: its prompts answered two before it", left, err)
	}
	if _, _, err := confirmInstruction(ctx, store, second.ID, protocol, "reward over base", reward, request.Base, reading(wins)); err == nil || !strings.Contains(err.Error(), "fresh partition") || readings != 3 {
		t.Fatalf("a further comparison under the replacement = %v, want it refused: the prompts have answered what they allow", err)
	}

	// Another prompt of each group stands for it: the groups are the ones that
	// answered, so an acceptance over the other prompts starts with them spent.
	stood := map[string]bool{}
	for _, shard := range []artifact.ID{partition.Development, partition.Selection, partition.Promotion, partition.Audit} {
		membership, _, err := dataset.LoadMembership(ctx, store, shard)
		if err != nil {
			t.Fatal(err)
		}
		for _, record := range membership.Records {
			stood[record.ID] = true
		}
	}
	others := request
	others.Cases = slices.DeleteFunc(slices.Clone(cases), func(eligible InstructionCase) bool { return stood[eligible.Record] })
	third, apart := publish(others)
	if apart.Promotion == partition.Promotion || third.Groups != acceptance.Groups {
		t.Fatalf("the acceptance over each group's other prompts holds part %s of %d groups, want another part of the same %d groups", apart.Promotion, third.Groups, acceptance.Groups)
	}
	if _, _, err := confirmInstruction(ctx, store, third.ID, protocol, "reward over base", reward, request.Base, reading(wins)); err == nil || !strings.Contains(err.Error(), "fresh partition") || readings != 3 {
		t.Fatalf("a comparison on other prompts of groups that have answered = %v with %d readings, want it refused unread", err, readings)
	}

	// By the group alone: a record no part holds, of a group that answered, and one of a group no part holds.
	answered, _, err := exposure(ctx, store, apart, artifact.ID{}, []dataset.Record{{ID: "a prompt no part holds", Group: promoted.Group}})
	if fresh, _, freshErr := exposure(ctx, store, apart, artifact.ID{}, []dataset.Record{{ID: "a prompt no part holds", Group: "a group no part holds"}}); err != nil || freshErr != nil || answered != 3 || fresh != 0 {
		t.Fatalf("a new prompt of a group that answered three comparisons has answered %d, and one of a new group %d (%v, %v); want three and none", answered, fresh, err, freshErr)
	}

	// A likelihood pool is still what it was, and neither kind confirms under the other's rule.
	pool := publishedPool(t, store, HeldoutPoolRequest{
		Objective: objective, Base: request.Base, Sequence: 8,
		Basis: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvaluation, "pool fixture verdict")}, Weights: [poolPartCount]uint64{1, 2, 4, 4},
	}, 2)
	if read, err := RequireHeldoutPool(ctx, store, pool.Acceptance.ID); err != nil || read.Acceptance.Measure != poolMeasure || read.Acceptance.Rule != poolRule {
		t.Fatalf("a likelihood pool's acceptance reads back %+v (%v), want its own measure and rule", read.Acceptance, err)
	}
	unread := func([]string) ([]float64, error) {
		readings++
		return nil, nil
	}
	if _, _, err := confirmInstruction(ctx, store, pool.Acceptance.ID, protocol, "reward over base", reward, request.Base, reading(wins)); err == nil || readings != 3 {
		t.Fatalf("a likelihood pool asked for the pilot's verdict = %v, want it refused unread", err)
	}
	if _, _, err := ConfirmOnPool(ctx, store, acceptance.ID, reward, unread); err == nil || readings != 3 {
		t.Fatalf("the pilot's acceptance asked for a likelihood verdict = %v, want it refused unread", err)
	}
}
