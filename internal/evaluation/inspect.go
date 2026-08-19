package evaluation

import (
	"encoding/json"
	"errors"
)

type FailedObservation struct {
	Name string
	Body json.RawMessage
}

func FailedObservations(mediaType, schema string, data []byte) ([]FailedObservation, error) {
	var report struct {
		Observations []json.RawMessage `json:"observations"`
	}
	if err := json.Unmarshal(data, &report); err != nil {
		return nil, err
	}
	failures := make([]FailedObservation, 0)
	for _, raw := range report.Observations {
		failed, name, err := failedObservation(mediaType, schema, raw)
		if err != nil {
			return nil, err
		}
		if failed {
			failures = append(failures, FailedObservation{Name: name, Body: append(json.RawMessage(nil), raw...)})
		}
	}
	return failures, nil
}

func failedObservation(mediaType, schema string, raw json.RawMessage) (bool, string, error) {
	switch {
	case mediaType == generatedAnswerReportMedia && schema == generatedAnswerReportSchema,
		mediaType == structuredGeneratedReportMedia && schema == structuredGeneratedReportSchema:
		var value struct {
			Name     string `json:"name"`
			Accepted bool   `json:"accepted"`
		}
		if err := json.Unmarshal(raw, &value); err != nil {
			return false, "", err
		}
		return !value.Accepted, value.Name, nil
	case mediaType == multipleChoiceReportMedia && schema == multipleChoiceReportSchema,
		mediaType == groupedChoiceReportMedia && schema == groupedChoiceReportSchema,
		mediaType == mmluProReportMedia && schema == mmluProReportSchema:
		var value struct {
			Name     string `json:"name"`
			Selected int    `json:"selected"`
			Answer   int    `json:"answer"`
		}
		if err := json.Unmarshal(raw, &value); err != nil {
			return false, "", err
		}
		return value.Selected != value.Answer, value.Name, nil
	case mediaType == instructionRulesReportMedia && schema == instructionRulesReportSchema:
		var value struct {
			Name   string `json:"name"`
			Strict []bool `json:"strict"`
		}
		if err := json.Unmarshal(raw, &value); err != nil {
			return false, "", err
		}
		return !allTrue(value.Strict), value.Name, nil
	case mediaType == shardReportMediaType && schema == shardReportSchema:
		var value struct {
			Passed bool `json:"passed"`
		}
		if err := json.Unmarshal(raw, &value); err != nil {
			return false, "", err
		}
		return !value.Passed, "", nil
	case mediaType == probabilityMassReportMedia && schema == probabilityMassReportSchema,
		mediaType == campaignReportMediaType && schema == campaignReportSchema:
		return false, "", nil
	default:
		return false, "", errors.New("evaluation: unsupported report contract")
	}
}
