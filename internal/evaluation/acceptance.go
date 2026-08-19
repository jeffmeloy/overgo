package evaluation

import (
	"errors"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

const (
	acceptancePolicyVersion uint16 = 1
	acceptancePolicyMedia          = "application/vnd.overgo.evaluation-acceptance+json"
	acceptancePolicySchema         = "overgo/evaluation-acceptance/v1"
)

type MetricContract struct {
	Name      string              `json:"name"`
	Unit      string              `json:"unit,omitempty"`
	Direction runrecord.Direction `json:"direction"`
}

type AcceptancePolicy struct {
	ID      artifact.ID      `json:"-"`
	Version uint16           `json:"version"`
	Metrics []MetricContract `json:"metrics"`
}

var acceptancePolicyCodec = artifact.JSONDocumentCodec(
	"evaluation acceptance policy", artifact.KindProfile, acceptancePolicyMedia, acceptancePolicySchema,
	canonicalizeAcceptancePolicy, func(value AcceptancePolicy) artifact.ID { return value.ID },
	func(value *AcceptancePolicy, id artifact.ID) { value.ID = id }, cloneAcceptancePolicy,
)

func newAcceptancePolicy(metrics []runrecord.Metric) (AcceptancePolicy, error) {
	contracts := make([]MetricContract, len(metrics))
	for index, metric := range metrics {
		contracts[index] = MetricContract{Name: metric.Name, Unit: metric.Unit, Direction: metric.Direction}
	}
	return acceptancePolicyCodec.New(AcceptancePolicy{Version: acceptancePolicyVersion, Metrics: contracts})
}

func (policy AcceptancePolicy) Content() (artifact.Content, error) {
	return acceptancePolicyCodec.Content(policy)
}

func (policy AcceptancePolicy) admits(metrics []runrecord.Metric) bool {
	if acceptancePolicyCodec.ValidateIdentity(policy) != nil || len(metrics) != len(policy.Metrics) {
		return false
	}
	metrics = slices.Clone(metrics)
	sort.Slice(metrics, func(i, j int) bool { return metrics[i].Name < metrics[j].Name })
	for index, contract := range policy.Metrics {
		metric := metrics[index]
		if metric.Name != contract.Name || metric.Unit != contract.Unit || metric.Direction != contract.Direction {
			return false
		}
	}
	return true
}

func canonicalizeAcceptancePolicy(policy *AcceptancePolicy) error {
	if policy == nil || policy.Version != acceptancePolicyVersion || len(policy.Metrics) == 0 {
		return errors.New("evaluation: invalid acceptance policy")
	}
	sort.Slice(policy.Metrics, func(i, j int) bool { return policy.Metrics[i].Name < policy.Metrics[j].Name })
	for index, metric := range policy.Metrics {
		if metric.Name == "" || !validMetricDirection(metric.Direction) || index > 0 && policy.Metrics[index-1].Name == metric.Name {
			return errors.New("evaluation: invalid acceptance metric")
		}
	}
	return nil
}

func cloneAcceptancePolicy(policy AcceptancePolicy) AcceptancePolicy {
	policy.Metrics = slices.Clone(policy.Metrics)
	return policy
}
