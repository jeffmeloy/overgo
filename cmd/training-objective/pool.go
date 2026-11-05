package main

import (
	"context"
	"fmt"
	"io"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/trainingworkflow"
)

// poolWeights reads the weights that divide a pool's units among the four parts, each of which must have one.
func poolWeights(text string) (weights [len(evaluation.PoolParts)]uint64, err error) {
	for field := range strings.SplitSeq(text, ",") {
		name, value, _ := strings.Cut(strings.TrimSpace(field), "=")
		weight, err := strconv.Atoi(value)
		if at := slices.Index(evaluation.PoolParts[:], name); err == nil && at >= 0 && weight > 0 {
			weights[at] = uint64(weight)
		}
	}
	if slices.Contains(weights[:], 0) {
		return weights, fmt.Errorf("training-objective: -pool-weights %q: want a positive weight for each of %s", text, strings.Join(evaluation.PoolParts[:], ", "))
	}
	return weights, nil
}

// buildPool publishes the held-out pool of a registered objective: its
// held-out documents divided into development, selection, promotion and
// audit, the acceptance a confirmation is judged by, and the budgets the
// promotion and audit parts grant. The gain the pool is sized to show is
// stated by whoever runs this, from the gains the objective's verdicts on
// record show, which the output lists beside it with the pool's sizes, the
// error rates, the confirmations the promotion part may answer and the
// smallest gain it can show at that count.
func buildPool(repository, alias string, gain float64, sequence uint64, weightsText string, output io.Writer) error {
	request := evaluation.HeldoutPoolRequest{Gain: gain, Sequence: sequence}
	var err error
	if request.Weights, err = poolWeights(weightsText); err != nil {
		return err
	}
	if gain <= 0 || sequence == 0 {
		return fmt.Errorf("training-objective: -pool-gain and -pool-seq are required: the gain the pool is sized to show, in standard deviations of the paired per-document difference, and the tokens each document is read as far as")
	}
	root, err := dataroot.StoreRoot(repository)
	if err != nil {
		return err
	}
	store, err := overgodb.Open(root)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	if request.Objective, err = trainingworkflow.LoadObjectiveAlias(ctx, store, alias); err != nil {
		return err
	}
	var recorded []float64
	if recorded, request.Base, request.Basis, err = evaluation.RecordedGains(ctx, store, request.Objective); err != nil {
		return err
	}
	// What was charged before a part named what it holds is found from this pool only once those parts are named.
	if _, err := evaluation.NameEarlierParts(ctx, store); err != nil {
		return err
	}
	pool, err := evaluation.BuildHeldoutPool(ctx, store, request)
	if err != nil {
		return err
	}
	if _, err := artifact.Publish(ctx, store, pool.Batch); err != nil {
		return err
	}
	acceptance := pool.Acceptance
	_, err = fmt.Fprintf(output, "pool of objective %s over base %s\n"+
		"%v documents in %v, each read as far as %d tokens\n"+
		"sized to show a gain of %.2f standard deviations of the paired difference; the verdicts on record show %.2f\n"+
		"a gain that is not there is reported at most %.2f of the time over every confirmation, and a gain of that size is missed at most %.2f\n"+
		"the promotion part answers %d confirmations, each passing a statistic of %.3f; the smallest gain it shows at that count is %.2f standard deviations; the audit part is read once\n"+
		"split %s\nacceptance %s\n",
		request.Objective.ID, request.Base, pool.Documents, evaluation.PoolParts, acceptance.Sequence, acceptance.Gain, recorded,
		acceptance.FalseGain, acceptance.MissedGain, acceptance.Confirmations, acceptance.Critical, acceptance.SmallestGain(),
		pool.Partition.ID, acceptance.ID)
	return err
}
