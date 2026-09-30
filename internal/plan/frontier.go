package plan

// Ref names one plan row as an exact item/step identity.
type Ref struct {
	Item string `json:"item"`
	Step string `json:"step"`
}

// String renders the canonical item/step reference.
func (r Ref) String() string { return r.Item + stepReferenceSeparator + r.Step }

// ReadyFrontier returns every dispatchable open step in deterministic
// document order: each open step whose dependencies are all proven by gated
// completion evidence or retained done state. Current remains the
// single-lane compatibility selector; its selection is always a member of
// this frontier, so a single-lane driver and a leased multi-lane driver can
// never disagree about what is dispatchable.
func ReadyFrontier(d Plan) ([]Ref, error) {
	frontier := []Ref{}
	for _, item := range d.Items {
		if item.Status != StatusOpen {
			continue
		}
		for _, step := range item.Steps {
			if step.Status != StatusOpen {
				continue
			}
			if dependenciesSatisfied(d, step) {
				frontier = append(frontier, Ref{Item: item.ID, Step: step.ID})
			}
		}
	}
	return frontier, nil
}

// A plan reference joins its item and step with a slash.
const stepReferenceSeparator = "/"
