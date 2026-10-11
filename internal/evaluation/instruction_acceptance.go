package evaluation

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/runrecord"
	"overgo/internal/tensorstats"
)

const (
	instructionAcceptanceMedia  = "application/vnd.overgo.instruction-acceptance+json"
	instructionAcceptanceSchema = "overgo/instruction-acceptance/v1"
	// InstructionScorer names the outcome the instruction pilot is judged by
	// and the version of the code that scores it: every rule of a prompt
	// passed, strictly, as instruction_rules.go reads an answer. A change to
	// what that code calls passed is another version, and an acceptance fixed
	// for an earlier one confirms nothing.
	InstructionScorer = "instruction-rules/strict/1"
	// instructionEven is the chance of a win on a pair the arms differ on when
	// neither is better.
	instructionEven = 0.5
)

// ErrInstructionEvidence refuses an acceptance whose promotion groups cannot
// show the gain at the stated rates: the pilot is then inconclusive before a
// training run is spent, and the refusal is its record.
var ErrInstructionEvidence = errors.New("evaluation: not enough evidence can be had")

// InstructionProtocol is how a prompt of the instruction pilot is asked and
// its answer scored: every setting an outcome depends on beside the weights.
// The tokenizer is the base model's, which the acceptance names.
type InstructionProtocol struct {
	Scorer    string `json:"scorer"`
	Template  string `json:"template"`
	Thinking  string `json:"thinking"`
	Sampling  string `json:"sampling"`
	Stop      string `json:"stop"`
	Window    uint64 `json:"window"`
	MaxTokens uint64 `json:"max_tokens"`
}

// InstructionComparison is one comparison of an acceptance's family, named
// before any arm is trained: the arm that is to be shown better and the arm it
// is compared with.
type InstructionComparison struct {
	Name      string `json:"name"`
	Candidate string `json:"candidate"`
	Baseline  string `json:"baseline"`
}

// InstructionAcceptance is what a confirmation of the instruction pilot is
// judged by, fixed before any arm is trained. Its unit is a group: the prompts
// that ask for one combination of rules are one group, one prompt of each
// stands for it, drawn by Seed before any answer is read, and each group
// weighs the same. A group is wholly in one part of Split, and Groups of them
// are its promotion part: the independent pairs of a comparison, whose outcome
// for an arm is whether its answer passed every rule. Other prompts and
// averages over prompts describe an arm and decide nothing.
//
// Gain is the least gain in the share of groups passed that is worth
// confirming, FalseGain the rate at which the whole family of Comparisons may
// report a gain that is not there, divided equally among them, and MissedGain
// the rate at which one may miss a gain of that size. Discordance is the
// largest share of pairs that arms a gain apart may differ on that the
// development readings in Basis allow, and Power the exact chance a comparison
// at that share shows a gain of Gain. No arm is trained when an acceptance is
// fixed, so what is read is the base, twice on each development group from
// its own distribution: two answers of one arm differ on a share of groups,
// whose upper bound at the false-gain rate is where arms of equal worth
// differ, and an arm that passes the gain more of the groups and loses none
// differs on at most the gain more. That an arm loses nowhere is an
// assumption: one that gains on some groups and loses on others differs on
// more, and its gain is shown with less chance than Power. So is it that one
// rule combination is one independent unit: groups a generator links are to
// be merged by whoever lists the cases. Training names the specification the
// arms are trained under.
type InstructionAcceptance struct {
	ID          artifact.ID             `json:"-"`
	Version     uint16                  `json:"version"`
	Split       artifact.ID             `json:"split"`
	Base        artifact.ID             `json:"base"`
	Training    artifact.ID             `json:"training"`
	Protocol    InstructionProtocol     `json:"protocol"`
	Seed        uint64                  `json:"seed"`
	Groups      uint64                  `json:"groups"`
	Gain        float64                 `json:"gain"`
	FalseGain   float64                 `json:"false_gain"`
	MissedGain  float64                 `json:"missed_gain"`
	Discordance float64                 `json:"discordance"`
	Power       float64                 `json:"power"`
	Comparisons []InstructionComparison `json:"comparisons"`
	Basis       []artifact.ID           `json:"basis"`
}

var instructionAcceptanceCodec = artifact.JSONDocumentCodec(
	"instruction acceptance", artifact.KindEvidence, instructionAcceptanceMedia, instructionAcceptanceSchema,
	func(value *InstructionAcceptance) error {
		protocol := value.Protocol
		if value.Version != artifact.InitialDocumentVersion || !value.Split.Valid() || !value.Base.Valid() || !value.Training.Valid() ||
			slices.Contains([]string{protocol.Scorer, protocol.Template, protocol.Thinking, protocol.Sampling, protocol.Stop}, "") ||
			protocol.Window == 0 || protocol.MaxTokens == 0 || value.Groups == 0 || value.Gain <= 0 || len(value.Comparisons) == 0 || len(value.Basis) == 0 {
			return errors.New("evaluation: an instruction acceptance fixes its split, base, training, protocol, groups, gain, comparisons and basis")
		}
		return nil
	},
	func(value InstructionAcceptance) artifact.ID { return value.ID },
	func(value *InstructionAcceptance, id artifact.ID) { value.ID = id },
	func(value InstructionAcceptance) InstructionAcceptance {
		value.Comparisons, value.Basis = slices.Clone(value.Comparisons), slices.Clone(value.Basis)
		return value
	},
)

// InstructionCase is one prompt eligible for the pilot: its record in the
// dataset, the names of the rules it asks for, and Content, a digest of what
// it asks that is the same wherever the case is registered. A part of an
// acceptance is published as holding its cases' contents, so that what a
// prompt has answered is found when the same benchmark is imported again as
// another dataset.
type InstructionCase struct {
	Record  string
	Rules   []string
	Content string
}

// InstructionAcceptanceRequest states what an instruction acceptance is fixed
// from: the dataset and its eligible cases, the base model, the training
// specification, the protocol, the seed that draws each group's prompt and
// divides the groups by Weights, the gain worth confirming, the comparisons,
// and the development readings: Pairs development groups whose prompt the
// base answered twice, Discordant of them groups whose two answers differ in
// passing every rule, recorded in Basis.
type InstructionAcceptanceRequest struct {
	Dataset, Base, Training artifact.ID
	Cases                   []InstructionCase
	Protocol                InstructionProtocol
	Seed                    uint64
	Weights                 [poolPartCount]uint64
	Gain                    float64
	Comparisons             []InstructionComparison
	Pairs, Discordant       uint64
	Basis                   []artifact.ID
}

// instructionPower is the exact chance that a comparison over groups
// independent pairs reads a gain at level, when a share discordance of pairs
// differ and the candidate passes a share gain more of them than the baseline:
// summed over every count of discordant pairs, the chance of that count times
// the chance of the wins the exact test passes at it.
func instructionPower(groups uint64, discordance, gain, level float64) (power float64) {
	win := instructionEven + instructionEven*gain/discordance
	for discordant := uint64(1); discordant <= groups; discordant++ {
		passing := discordant + 1
		for passing > 0 && tensorstats.ExactPairedBinary(passing-1, discordant-passing+1, 0).Tail <= level {
			passing--
		}
		count := tensorstats.BinomialUpperTail(groups, discordant, discordance) - tensorstats.BinomialUpperTail(groups, discordant+1, discordance)
		power += count * tensorstats.BinomialUpperTail(discordant, passing, win)
	}
	return power
}

// divideInstructionGroups groups a request's cases and divides the groups
// among the four parts. Each combination of rules is a group, a repeated rule
// counted once; the prompt that stands for a group is the one a digest of the
// seed and its record puts first, which no outcome enters.
func divideInstructionGroups(request InstructionAcceptanceRequest) (dataset.SplitPlan, error) {
	stands := map[string]string{}
	for _, eligible := range request.Cases {
		rules := make([]string, len(eligible.Rules))
		for index, rule := range eligible.Rules {
			rules[index], _, _ = strings.Cut(rule, "/")
		}
		slices.Sort(rules)
		group := strings.Join(slices.Compact(rules), "+")
		drawn := func(record string) string {
			digest := sha256.Sum256(fmt.Appendf(nil, "%d/%s", request.Seed, record))
			return string(digest[:])
		}
		if held, found := stands[group]; !found || drawn(eligible.Record) < drawn(held) {
			stands[group] = eligible.Record
		}
	}
	records := make([]dataset.Record, 0, len(stands))
	for group, record := range stands {
		records = append(records, dataset.Record{ID: record, Group: group})
	}
	weights := make([]dataset.SplitPartition, len(PoolParts))
	for index, name := range PoolParts {
		weights[index] = dataset.SplitPartition{Name: name, Weight: request.Weights[index]}
	}
	plan, err := dataset.BuildGroupSplit(request.Dataset, records, request.Seed, weights)
	if err != nil {
		return dataset.SplitPlan{}, fmt.Errorf("evaluation: the pilot's %d groups of %d cases: %w", len(records), len(request.Cases), err)
	}
	return plan, nil
}

// InstructionGroups names, for each part of the split a request's acceptance
// would fix, the prompts that stand for the part's groups, in order. It reads
// no answer and needs no development reading, so the development part's
// prompts can be read before the acceptance is fixed from what they answer.
func InstructionGroups(request InstructionAcceptanceRequest) ([poolPartCount][]string, error) {
	var parts [poolPartCount][]string
	plan, err := divideInstructionGroups(request)
	for _, membership := range plan.Memberships {
		at := slices.Index(PoolParts[:], membership.Partition)
		for _, record := range membership.Records {
			parts[at] = append(parts[at], record.ID)
		}
		slices.Sort(parts[at])
	}
	return parts, err
}

// BuildInstructionAcceptance fixes an instruction acceptance and the batch
// that publishes it with its partition and the budget its promotion part
// grants, one confirmation a comparison. Its groups are divided as
// InstructionGroups names them. A request that leaves a gain, a rate's basis,
// a protocol setting or the training specification unnamed is refused, and so
// is one whose promotion groups cannot show the gain at the stated rates: the
// acceptance is not made larger than its independent groups to pass.
func BuildInstructionAcceptance(request InstructionAcceptanceRequest) (InstructionAcceptance, artifact.Batch, error) {
	plan, err := divideInstructionGroups(request)
	if err != nil {
		return InstructionAcceptance{}, artifact.Batch{}, err
	}
	var shards [poolPartCount]artifact.ID
	var groups uint64
	for _, membership := range plan.Memberships {
		at := slices.Index(PoolParts[:], membership.Partition)
		if shards[at] = membership.ID; at == poolPromotion {
			groups = uint64(len(membership.Records))
		}
	}
	proposer, err := changeProposer(request.Training)
	if err != nil {
		return InstructionAcceptance{}, artifact.Batch{}, err
	}
	partition, err := runrecord.NewSplitPartition(request.Dataset, shards[poolDevelopment], shards[poolSelection], shards[poolPromotion], shards[poolAudit], proposer)
	if err != nil {
		return InstructionAcceptance{}, artifact.Batch{}, err
	}
	// Arms a gain apart differ where two readings of the base do and, at most, on the gain more.
	_, repeated := tensorstats.ExactShareBounds(request.Discordant, request.Pairs, falseGain)
	discordance := min(repeated+request.Gain, 1)
	acceptance := InstructionAcceptance{
		Split: partition.ID, Base: request.Base, Training: request.Training, Protocol: request.Protocol, Seed: request.Seed, Groups: groups,
		Gain: request.Gain, FalseGain: falseGain, MissedGain: missedGain, Discordance: discordance,
		Comparisons: slices.Clone(request.Comparisons), Basis: slices.Clone(request.Basis),
	}
	if request.Pairs == 0 || len(request.Comparisons) == 0 || request.Gain <= 0 || request.Gain >= 1 {
		return InstructionAcceptance{}, artifact.Batch{}, fmt.Errorf("evaluation: an instruction acceptance is sized from its comparisons, a gain worth confirming and development pairs; %d comparisons, a gain of %.3f and %d of %d groups read twice differing, at most %.3f of them, size nothing",
			len(request.Comparisons), request.Gain, request.Discordant, request.Pairs, repeated)
	}
	acceptance.Power = instructionPower(groups, discordance, request.Gain, falseGain/float64(len(request.Comparisons)))
	if acceptance.Power < 1-missedGain {
		return InstructionAcceptance{}, artifact.Batch{}, fmt.Errorf("%w: %d independent groups in the promotion part show a gain of %.3f with chance %.3f where %.2f is wanted, at %.3f of pairs discordant and %d comparisons; more groups, not more prompts of these, would show it",
			ErrInstructionEvidence, groups, request.Gain, acceptance.Power, 1-missedGain, discordance, len(request.Comparisons))
	}
	if acceptance, err = instructionAcceptanceCodec.NewInitial(acceptance); err != nil {
		return InstructionAcceptance{}, artifact.Batch{}, err
	}
	key := poolKey + acceptance.ID.String()
	batch, err := plan.PublicationBatch(key, nil)
	if err != nil {
		return InstructionAcceptance{}, artifact.Batch{}, err
	}
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: proposer})
	// The held-out parts are published as holding what their cases ask, so a
	// case listed without it leaves a part that cannot be followed.
	contents := map[string]string{}
	for _, eligible := range request.Cases {
		if contents[eligible.Record] = eligible.Content; eligible.Content == "" {
			return InstructionAcceptance{}, artifact.Batch{}, fmt.Errorf("evaluation: case %q is listed without the content it asks; an instruction acceptance names it, so that what a prompt has answered is found wherever the case is registered", eligible.Record)
		}
	}
	if err := containContents(&batch, plan.Memberships, contents); err != nil {
		return InstructionAcceptance{}, artifact.Batch{}, err
	}
	granted, err := runrecord.NewBudget(poolUnit, partition.Promotion, uint64(len(acceptance.Comparisons)), acceptance.ID)
	if err != nil {
		return InstructionAcceptance{}, artifact.Batch{}, err
	}
	for _, stage := range []func(string) (artifact.Batch, error){partition.Batch, granted.Batch, func(key string) (artifact.Batch, error) {
		return instructionAcceptanceCodec.Batch(key, acceptance, artifact.DependencyLineage(acceptance.ID, partition.ID, request.Training), nil)
	}} {
		part, err := stage(key)
		if err != nil {
			return InstructionAcceptance{}, artifact.Batch{}, err
		}
		batch.Contents, batch.Lineage = append(batch.Contents, part.Contents...), append(batch.Lineage, part.Lineage...)
	}
	return acceptance, batch, nil
}

// RequireInstructionAcceptance reads a published instruction acceptance and
// the prompts that stand for the groups of each part of its split, in order,
// for a caller whose instruction cases are the records of the dataset cases.
// An acceptance whose split divides another dataset is refused: its groups
// are named by record, and under the same names another dataset's prompts and
// rules may be other ones.
func RequireInstructionAcceptance(ctx context.Context, reader artifact.Reader, id, cases artifact.ID) (InstructionAcceptance, [poolPartCount][]string, error) {
	var parts [poolPartCount][]string
	acceptance, err := instructionAcceptanceCodec.Require(ctx, reader, id)
	if err != nil {
		return InstructionAcceptance{}, parts, err
	}
	content, found, err := artifact.ReadContent(ctx, reader, acceptance.Split)
	if err != nil || !found {
		return InstructionAcceptance{}, parts, errors.Join(fmt.Errorf("evaluation: the partition of acceptance %s is not committed", acceptance.ID), err)
	}
	partition, err := runrecord.ParseSplitPartition(content.Data)
	if err != nil {
		return InstructionAcceptance{}, parts, err
	}
	if partition.Dataset != cases {
		return InstructionAcceptance{}, parts, fmt.Errorf("evaluation: acceptance %s divides the instruction cases of dataset %s, and the cases at hand are of dataset %s; an acceptance is run on the prompts and rules it was fixed over", acceptance.ID, partition.Dataset, cases)
	}
	for index, shard := range []artifact.ID{partition.Development, partition.Selection, partition.Promotion, partition.Audit} {
		if parts[index], err = partDocuments(ctx, reader, shard); err != nil {
			return InstructionAcceptance{}, parts, err
		}
		slices.Sort(parts[index])
	}
	return acceptance, parts, nil
}

// Gained says whether a confirmation's verdict is a gain: the one verdict a
// comparison is adopted on. A loss and not enough evidence are both no gain.
func (confirmation PoolConfirmation) Gained() bool { return confirmation.Verdict == poolGain }

// ConfirmInstruction has the promotion part of an instruction acceptance
// answer one comparison of its family. The comparison is one the acceptance
// named, the protocol the answers were made under is the acceptance's and the
// scorer this code's, and no group is in two parts of the split: otherwise it
// is refused before anything is charged. A comparison is charged once, for
// the candidate and the baseline it first names, before the part's prompts are
// handed to read, which returns for each whether the candidate's answer passed
// every rule and whether the baseline's did. The verdict is the exact test of
// the pairs the arms differ on at the comparison's share of the family's
// false-gain rate, and is all that is published of the reading. What the
// part's prompts have answered under any acceptance is spent, as a pool's is
// (confirmOnPart).
func ConfirmInstruction(ctx context.Context, store artifact.Repository, acceptanceID artifact.ID, protocol InstructionProtocol, comparison string, candidate, baseline artifact.ID,
	read func(prompts []string) (passed, base []bool, err error)) (PoolConfirmation, uint64, error) {
	return oneDecision(store, func(head artifact.CommitID) (PoolConfirmation, uint64, error) {
		acceptance, err := instructionAcceptanceCodec.Require(ctx, store, acceptanceID)
		if err != nil {
			return PoolConfirmation{}, 0, err
		}
		issued := uint64(len(acceptance.Comparisons))
		named, purpose := "comparison "+comparison+": ", candidate.String()+" over "+baseline.String()
		switch {
		case !slices.ContainsFunc(acceptance.Comparisons, func(fixed InstructionComparison) bool { return fixed.Name == comparison }):
			return PoolConfirmation{}, 0, fmt.Errorf("evaluation: acceptance %s named its %d comparisons before training, and %q is not one of them; a further comparison is a further acceptance", acceptance.ID, issued, comparison)
		case protocol != acceptance.Protocol || protocol.Scorer != InstructionScorer:
			return PoolConfirmation{}, 0, fmt.Errorf("evaluation: acceptance %s fixed the protocol %+v, and the answers were made under %+v and scored by %s; nothing was charged", acceptance.ID, acceptance.Protocol, protocol, InstructionScorer)
		case !candidate.Valid() || !baseline.Valid():
			return PoolConfirmation{}, 0, errors.New("evaluation: a comparison names its candidate and its baseline before it reads anything")
		}
		content, found, err := artifact.ReadContent(ctx, store, acceptance.Split)
		if err != nil || !found {
			return PoolConfirmation{}, 0, errors.Join(fmt.Errorf("evaluation: the partition of acceptance %s is not committed", acceptance.ID), err)
		}
		partition, err := runrecord.ParseSplitPartition(content.Data)
		if err != nil {
			return PoolConfirmation{}, 0, err
		}
		// A record or a group that two parts hold is a group divided between them.
		held := map[string]int{}
		for part, shard := range []artifact.ID{partition.Development, partition.Selection, partition.Promotion, partition.Audit} {
			membership, found, err := dataset.LoadMembership(ctx, store, shard)
			if err != nil || !found {
				return PoolConfirmation{}, 0, errors.Join(fmt.Errorf("evaluation: the pool's part %s is not committed", shard), err)
			}
			for _, record := range membership.Records {
				for _, name := range []string{record.ID, record.Group} {
					if other, found := held[name]; found && other != part {
						return PoolConfirmation{}, 0, fmt.Errorf("evaluation: %q is in the %s and the %s part of acceptance %s; a group is wholly in one, and nothing was charged", name, PoolParts[other], PoolParts[part], acceptance.ID)
					}
					held[name] = part
				}
			}
		}
		granted, err := runrecord.NewBudget(poolUnit, partition.Promotion, issued, acceptance.ID)
		if err != nil {
			return PoolConfirmation{}, 0, err
		}
		_, charges, err := runrecord.BudgetCharges(ctx, store, granted.ID)
		if err != nil {
			return PoolConfirmation{}, 0, err
		}
		if slices.ContainsFunc(charges, func(charge runrecord.BudgetCharge) bool {
			return strings.HasPrefix(charge.Purpose, named) && charge.Purpose != named+purpose
		}) {
			return PoolConfirmation{}, 0, fmt.Errorf("evaluation: comparison %q of acceptance %s was charged for another candidate or baseline; one of each is fixed before the promotion part is opened", comparison, acceptance.ID)
		}
		level := acceptance.FalseGain / float64(issued)
		return confirmOnPart(ctx, store, head, partition, acceptance.ID, acceptance.Training, candidate, acceptance.Groups, issued, named+purpose, func(prompts []string) (string, error) {
			passed, base, err := read(prompts)
			if err != nil || len(passed) != len(prompts) || len(base) != len(prompts) {
				return "", errors.Join(fmt.Errorf("evaluation: the arms were read on %d and %d of the promotion part's %d groups; the comparison is charged and its verdict is owed", len(passed), len(base), len(prompts)), err)
			}
			var wins, losses uint64
			for index := range prompts {
				switch {
				case passed[index] && !base[index]:
					wins++
				case base[index] && !passed[index]:
					losses++
				}
			}
			ties := uint64(len(prompts)) - wins - losses
			switch {
			case tensorstats.ExactPairedBinary(wins, losses, ties).Tail <= level:
				return poolGain, nil
			case tensorstats.ExactPairedBinary(losses, wins, ties).Tail <= level:
				return poolLoss, nil
			}
			return poolUndecided, nil
		})
	})
}
