package evaluation

import "encoding/json"

// NormalizeEvaluationPlan admits field-order variants at an external boundary
// and returns the exact native RepoDB identity bytes.
func NormalizeEvaluationPlan(data []byte) (Plan, []byte, error) {
	plan, err := ParsePlan(data)
	if err != nil {
		return Plan{}, nil, err
	}
	canonical, err := json.Marshal(plan.body)
	if err != nil {
		return Plan{}, nil, err
	}
	return plan, canonical, nil
}

// NormalizeEvaluationEvidence admits external serialization only through the
// native evidence codec; foreign identities and schemas remain non-authority.
func NormalizeEvaluationEvidence(data []byte) (EvaluationEvidence, []byte, error) {
	return evaluationEvidenceCodec.Normalize(data)
}
