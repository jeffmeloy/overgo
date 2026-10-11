package evaluation

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/trainingprogram"
)

const (
	poolAcceptanceMedia  = "application/vnd.overgo.heldout-acceptance+json"
	poolAcceptanceSchema = "overgo/heldout-acceptance/v1"

	// falseGain is the chance, over every confirmation a pool answers, that one
	// reports a gain that is not there. One in twenty is the level a single
	// held-out comparison is read at; it is held for the whole family of
	// confirmations, so that asking the pool more often makes each answer
	// stricter and not the family looser.
	falseGain = 0.05
	// missedGain is the chance a confirmation misses a gain of the size the
	// pool is sized for. A missed gain costs one more run; a false one moves
	// the base every later change is compared with, so the two are not held
	// to one rate.
	missedGain = 0.2

	// poolMeasure is what a confirmation reads: for each document, the mean
	// log-likelihood of its tokens under the candidate less that under the
	// base. Documents are the unit, since the tokens of one document are not
	// independent of each other.
	poolMeasure = "mean token log-likelihood per document, candidate less base"
	// poolRule is how a confirmation is decided and how many the pool may
	// answer. A proposer that is told each verdict and nothing else of a
	// reading can still steer by verdicts, so the false-gain rate is divided
	// over every confirmation any sequence of verdicts could lead to.
	poolRule = "one-sided paired z; the false-gain rate divided over every sequence of three-way verdicts"
	// poolUnit is what a pool's budget counts.
	poolUnit = "confirmation"
	// poolVerdicts is the answers a confirmation has: a gain, a loss, or not
	// enough evidence.
	poolVerdicts = 3
)

// The four parts of a pool, in the order a split partition lists them.
const (
	poolDevelopment = iota
	poolSelection
	poolPromotion
	poolAudit
	poolPartCount
)

// PoolParts names the four parts, in that order.
var PoolParts = [poolPartCount]string{poolDevelopment: "development", poolSelection: "selection", poolPromotion: "promotion", poolAudit: "audit"}

// PoolAcceptance is what a confirmation on a held-out pool is judged by,
// fixed before any is read: the objective and the base its candidates are
// compared with, the measure and the rule, the gain the pool is sized to
// show, in standard deviations of the measure's paired difference, the two
// error rates, how many confirmations the promotion part may answer and the
// statistic each must pass, the verdicts the gain was stated from, and the
// tokens each document is read as far as by whoever ranks or confirms on it.
type PoolAcceptance struct {
	ID            artifact.ID   `json:"-"`
	Version       uint16        `json:"version"`
	Split         artifact.ID   `json:"split"`
	Objective     artifact.ID   `json:"objective"`
	Base          artifact.ID   `json:"base"`
	Measure       string        `json:"measure"`
	Rule          string        `json:"rule"`
	Gain          float64       `json:"gain"`
	FalseGain     float64       `json:"false_gain"`
	MissedGain    float64       `json:"missed_gain"`
	Documents     uint64        `json:"documents"`
	Confirmations uint64        `json:"confirmations"`
	Critical      float64       `json:"critical"`
	Basis         []artifact.ID `json:"basis"`
	// Sequence is omitted at zero: an acceptance published before the window
	// was fixed here names none, and nothing is ranked or confirmed on it.
	Sequence uint64 `json:"sequence,omitzero"`
}

var poolAcceptanceCodec = artifact.JSONDocumentCodec(
	"held-out pool acceptance", artifact.KindEvidence, poolAcceptanceMedia, poolAcceptanceSchema,
	func(value *PoolAcceptance) error {
		if value == nil || value.Version != artifact.InitialDocumentVersion || !value.Split.Valid() || !value.Objective.Valid() || !value.Base.Valid() ||
			value.Gain <= 0 || value.Documents == 0 || value.Confirmations == 0 || len(value.Basis) == 0 ||
			value.Critical != poolCritical(value.Confirmations) || poolDocuments(value.Gain, value.Confirmations) > value.Documents {
			return errors.New("evaluation: invalid held-out pool acceptance")
		}
		return nil
	},
	func(value PoolAcceptance) artifact.ID { return value.ID },
	func(value *PoolAcceptance, id artifact.ID) { value.ID = id },
	func(value PoolAcceptance) PoolAcceptance {
		value.Basis = slices.Clone(value.Basis)
		return value
	},
)

// SmallestGain is the smallest gain, in standard deviations of the paired
// difference, the promotion part shows at the two error rates when it answers
// every confirmation it may.
func (acceptance PoolAcceptance) SmallestGain() float64 {
	return (acceptance.Critical + normalQuantile(1-acceptance.MissedGain)) / math.Sqrt(float64(acceptance.Documents))
}

// normalQuantile is the value a standard normal falls below with probability p.
func normalQuantile(p float64) float64 { return math.Sqrt2 * math.Erfinv(p+p-1) }

// poolCritical is the statistic each confirmation must pass when the pool
// answers confirmations of them. A proposer told only verdicts can reach, by
// the last of them, any of 1 + 3 + 9 + ... candidates, one for every sequence
// of earlier verdicts; the false-gain rate is divided over all of them.
func poolCritical(confirmations uint64) float64 {
	reachable := (math.Pow(poolVerdicts, float64(confirmations)) - 1) / (poolVerdicts - 1)
	return normalQuantile(1 - falseGain/reachable)
}

// poolDocuments is the documents a part needs to show, when it answers
// confirmations of them, a gain of gain standard deviations of the paired
// difference at the two error rates.
func poolDocuments(gain float64, confirmations uint64) uint64 {
	root := (poolCritical(confirmations) + normalQuantile(1-missedGain)) / gain
	return uint64(math.Ceil(root * root))
}

// poolConfirmations is the most confirmations a part of documents can answer
// for the gain, none when it cannot answer one.
func poolConfirmations(documents uint64, gain float64) (confirmations uint64) {
	for poolDocuments(gain, confirmations+1) <= documents {
		confirmations++
	}
	return confirmations
}

// HeldoutPoolRequest states what a pool is built for: the objective whose
// held-out documents it divides, the base model its candidates are compared
// with, the gain it is sized to show, the verdicts that gain was stated from,
// the tokens each document is read as far as, and the weights that divide the
// documents no verdict has read among development, selection, promotion and
// audit.
type HeldoutPoolRequest struct {
	Objective trainingprogram.ObjectiveDocument
	Base      artifact.ID
	Gain      float64
	Basis     []artifact.ID
	Sequence  uint64
	Weights   [poolPartCount]uint64
}

// HeldoutPool is a pool as built: the split partition that binds its four
// parts, the acceptance a confirmation is judged by, the documents of each
// part, and the batch that publishes them with the budgets the promotion and
// audit parts grant.
type HeldoutPool struct {
	Partition  runrecord.SplitPartition
	Acceptance PoolAcceptance
	Documents  [poolPartCount]int
	Batch      artifact.Batch
}

// changeProposer names the proposer of changes to what an objective trains,
// the consumer a pool's promotion and audit parts are blinded from.
func changeProposer(objective artifact.ID) (artifact.ID, error) {
	return artifact.IdentifyBytes(artifact.KindEvidence, []byte("overgo/change-proposer/"+objective.String()))
}

// BuildHeldoutPool divides an objective's held-out documents, which its
// training membership does not hold, into the four parts a split partition
// has, by document and by the declared weights. The documents the
// objective's own verdicts score, the first by record identity, have been
// read: they stay what they are, the objective's regression check, and are in
// no part. A document whose content a training document, a read one or an
// earlier one of the pool already has is left out. The promotion part's size
// decides how many confirmations it may answer for the stated gain, the audit
// part is read once, and a pool whose promotion or audit part cannot show the
// gain once is refused with the number of documents each would need.
func BuildHeldoutPool(ctx context.Context, reader artifact.Reader, request HeldoutPoolRequest) (HeldoutPool, error) {
	objective := request.Objective
	if request.Sequence == 0 {
		return HeldoutPool{}, errors.New("evaluation: a pool fixes the tokens each document is read as far as before any document is read")
	}
	training, found, err := dataset.LoadMembership(ctx, reader, objective.Split)
	if err != nil || !found {
		return HeldoutPool{}, errors.Join(errors.New("evaluation: the objective's training membership is not committed"), err)
	}
	heldout, err := dataset.SplitPartner(ctx, reader, objective.Split)
	if err != nil {
		return HeldoutPool{}, err
	}
	contents, err := recordContents(ctx, reader, objective.Dataset)
	if err != nil {
		return HeldoutPool{}, err
	}
	read := len(heldout.Records)
	if scored := int(objective.HeldoutRecords); scored != 0 {
		read = min(read, scored)
	}
	seen := map[string]bool{}
	for _, record := range slices.Concat(training.Records, heldout.Records[:read]) {
		seen[contents[record.ID]] = true
	}
	var unread []dataset.Record
	for _, record := range heldout.Records[read:] {
		// A record with no recorded content is no copy of anything.
		if content := contents[record.ID]; content == "" || !seen[content] {
			seen[content] = true
			unread = append(unread, record)
		}
	}
	seed := sha256.Sum256([]byte(objective.ID.String()))
	weights := make([]dataset.SplitPartition, len(PoolParts))
	for index, name := range PoolParts {
		weights[index] = dataset.SplitPartition{Name: name, Weight: request.Weights[index]}
	}
	plan, err := dataset.BuildGroupSplit(objective.Dataset, unread, binary.BigEndian.Uint64(seed[:]), weights)
	if err != nil {
		return HeldoutPool{}, fmt.Errorf("evaluation: the objective's held-out membership leaves %d documents no verdict has read: %w", len(unread), err)
	}
	pool := HeldoutPool{}
	var shards [poolPartCount]artifact.ID
	for _, membership := range plan.Memberships {
		at := slices.Index(PoolParts[:], membership.Partition)
		shards[at], pool.Documents[at] = membership.ID, len(membership.Records)
	}
	promotion, audit := uint64(pool.Documents[poolPromotion]), uint64(pool.Documents[poolAudit])
	confirmations := poolConfirmations(promotion, request.Gain)
	if once := poolDocuments(request.Gain, 1); confirmations == 0 || audit < once {
		return HeldoutPool{}, fmt.Errorf("evaluation: a gain of %.2f standard deviations needs %d documents to be shown once, and the pool's promotion part holds %d and its audit part %d; widen the pool or state a larger gain",
			request.Gain, once, promotion, audit)
	}
	proposer, err := changeProposer(objective.ID)
	if err != nil {
		return HeldoutPool{}, err
	}
	if pool.Partition, err = runrecord.NewSplitPartition(objective.Dataset, shards[poolDevelopment], shards[poolSelection], shards[poolPromotion], shards[poolAudit], proposer); err != nil {
		return HeldoutPool{}, err
	}
	if pool.Acceptance, err = poolAcceptanceCodec.NewInitial(PoolAcceptance{
		Split: pool.Partition.ID, Objective: objective.ID, Base: request.Base, Measure: poolMeasure, Rule: poolRule,
		Gain: request.Gain, FalseGain: falseGain, MissedGain: missedGain,
		Documents: promotion, Confirmations: confirmations, Critical: poolCritical(confirmations), Basis: slices.Clone(request.Basis),
		Sequence: request.Sequence,
	}); err != nil {
		return HeldoutPool{}, err
	}
	key := poolKey + pool.Acceptance.ID.String()
	if pool.Batch, err = plan.PublicationBatch(key, nil); err != nil {
		return HeldoutPool{}, err
	}
	pool.Batch.Artifacts = append(pool.Batch.Artifacts, artifact.Descriptor{ID: proposer})
	stages := []func(string) (artifact.Batch, error){pool.Partition.Batch, func(key string) (artifact.Batch, error) {
		return poolAcceptanceCodec.Batch(key, pool.Acceptance, artifact.DependencyLineage(pool.Acceptance.ID, pool.Partition.ID, objective.ID), nil)
	}}
	// The promotion part answers the confirmations its size allows; the audit
	// part is read once, at a campaign's end.
	for _, grant := range []struct {
		shard  artifact.ID
		issued uint64
	}{{pool.Partition.Promotion, confirmations}, {pool.Partition.Audit, 1}} {
		budget, err := runrecord.NewBudget(poolUnit, grant.shard, grant.issued, pool.Acceptance.ID)
		if err != nil {
			return HeldoutPool{}, err
		}
		stages = append(stages, budget.Batch)
	}
	for _, stage := range stages {
		part, err := stage(key)
		if err != nil {
			return HeldoutPool{}, err
		}
		pool.Batch.Contents = append(pool.Batch.Contents, part.Contents...)
		pool.Batch.Lineage = append(pool.Batch.Lineage, part.Lineage...)
	}
	return pool, nil
}

// recordContents names the content of each record of a dataset whose assets
// are inventories of files, by the digest registration recorded for the file;
// a record of another kind of asset has none recorded and is named by nothing.
func recordContents(ctx context.Context, reader artifact.Reader, id artifact.ID) (map[string]string, error) {
	document, found, err := dataset.Load(ctx, reader, id)
	if err != nil || !found {
		return nil, errors.Join(fmt.Errorf("evaluation: dataset %s is not committed", id), err)
	}
	contents := map[string]string{}
	for _, asset := range document.Assets {
		inventory, found, err := dataset.LoadInventory(ctx, reader, asset.Artifact)
		if err != nil {
			return nil, err
		}
		if !found {
			continue
		}
		for index, file := range inventory.Files {
			contents[dataset.RecordID(id, asset.Name, uint64(index))] = file.Digest
		}
	}
	return contents, nil
}

// RecordedGains reads what the held-out verdicts on record show for an
// objective's contract: each verdict's gain as a multiple of the standard
// deviation of its paired record differences, in the metric's own direction,
// the base model they all judged, and the verdicts themselves. It is what a
// pool's gain is stated from and what its acceptance names as its basis.
// Verdicts of two base models, or none, are refused: a pool has one base.
func RecordedGains(ctx context.Context, store *overgodb.Store, objective trainingprogram.ObjectiveDocument) (gains []float64, base artifact.ID, basis []artifact.ID, err error) {
	verdicts, _, err := PastHeldoutVerdicts(ctx, store, 0, 0)
	if err != nil {
		return nil, artifact.ID{}, nil, err
	}
	for _, id := range verdicts {
		facts, err := ReadHeldoutVerdict(ctx, store, id)
		if err != nil {
			continue
		}
		judged, err := trainingprogram.LoadObjective(ctx, store, facts.Objective)
		if err != nil || !trainingprogram.SameContract(judged, objective) || len(facts.BaselineScores) != len(facts.CandidateScores) {
			continue
		}
		observation, err := runrecord.RequireServingObservation(ctx, store, facts.Observation)
		if err != nil {
			return nil, artifact.ID{}, nil, err
		}
		if base.Valid() && observation.Model != base {
			return nil, artifact.ID{}, nil, fmt.Errorf("evaluation: the objective's verdicts judge two base models, %s and %s", base, observation.Model)
		}
		differences := make([]RecordScore, len(facts.CandidateScores))
		var mean float64
		for index, score := range facts.CandidateScores {
			differences[index].Value = score.Value - facts.BaselineScores[index].Value
			mean += differences[index].Value / float64(len(differences))
		}
		if facts.Direction == runrecord.DirectionMinimize {
			mean = -mean
		}
		// A verdict whose records did not move shows no gain, whatever their spread.
		if mean != 0 {
			mean /= meanError(differences) * math.Sqrt(float64(len(differences)))
		}
		base, basis, gains = observation.Model, append(basis, id), append(gains, mean)
	}
	if len(basis) == 0 {
		return nil, artifact.ID{}, nil, errors.New("evaluation: no held-out verdict of this objective is on record to state a gain from")
	}
	return gains, base, basis, nil
}
