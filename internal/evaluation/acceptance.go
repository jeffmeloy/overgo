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
	return acceptancePolicyCodec.ValidateIdentity(policy) == nil && metricContractAdmits(policy.Metrics, metrics)
}

func canonicalizeAcceptancePolicy(policy *AcceptancePolicy) error {
	if policy == nil || policy.Version != acceptancePolicyVersion || len(policy.Metrics) == 0 {
		return errors.New("evaluation: invalid acceptance policy")
	}
	if !canonicalizeMetricContracts(policy.Metrics) {
		return errors.New("evaluation: invalid acceptance metric")
	}
	return nil
}

func canonicalizeMetricContracts(metrics []MetricContract) bool {
	sort.Slice(metrics, func(i, j int) bool { return metrics[i].Name < metrics[j].Name })
	for index, metric := range metrics {
		if metric.Name == "" || !validMetricDirection(metric.Direction) || index > 0 && metrics[index-1].Name == metric.Name {
			return false
		}
	}
	return true
}

func metricContractAdmits(contract []MetricContract, metrics []runrecord.Metric) bool {
	if len(contract) != len(metrics) {
		return false
	}
	metrics = slices.Clone(metrics)
	sort.Slice(metrics, func(i, j int) bool { return metrics[i].Name < metrics[j].Name })
	return orderedMetricContractAdmits(contract, metrics)
}

func orderedMetricContractAdmits(contract []MetricContract, metrics []runrecord.Metric) bool {
	for index, expected := range contract {
		actual := metrics[index]
		if actual.Name != expected.Name || actual.Unit != expected.Unit || actual.Direction != expected.Direction {
			return false
		}
	}
	return true
}

func cloneAcceptancePolicy(policy AcceptancePolicy) AcceptancePolicy {
	policy.Metrics = slices.Clone(policy.Metrics)
	return policy
}
