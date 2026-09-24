package webuilane

import (
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

const (
	// StylesheetPath is the workbench stylesheet, the only source of its design tokens.
	StylesheetPath = "internal/server/webui/style.css"
	// DesignDocumentPath is the document derived from the stylesheet.
	DesignDocumentPath = "docs/workbench_design.json"
)

// The text tokens and the opaque surfaces they may be read on, every pair of
// which is measured; the pairs the stylesheet paints elsewhere (the accent's
// text on the accent, ink on the translucent highlight) are measured beside.
var (
	textTokens    = []string{"ink", "dim", "faint", "acc", "amber", "ok", "err"}
	surfaceTokens = []string{"bg0", "bg1", "bg2", "bg3"}
)

// DesignDocument is the workbench's design as style.css declares it: each
// colour scheme's colour tokens with the contrast of every text token on
// every surface and of every pair the stylesheet paints, and the spacing,
// type and radius scales.
type DesignDocument struct {
	Source  string            `json:"source"`
	Schemes []DesignScheme    `json:"schemes"`
	Spacing map[string]string `json:"spacing"`
	Type    map[string]string `json:"type"`
	Radius  map[string]string `json:"radius"`
}

// DesignScheme is one colour scheme's tokens and measured pairs.
type DesignScheme struct {
	Name     string            `json:"name"`
	Colours  map[string]string `json:"colours"`
	Contrast []TokenContrast   `json:"contrast"`
}

// TokenContrast is one text token read on one surface.
type TokenContrast struct {
	Foreground string  `json:"foreground"`
	Background string  `json:"background"`
	Ratio      float64 `json:"ratio"`
}

var (
	rootBlock      = regexp.MustCompile(`(?s)(?:^|\n):root \{(.*?)\n\}`)
	lightRootBlock = regexp.MustCompile(`(?s)@media \(prefers-color-scheme: light\) \{\s*:root \{(.*?)\n  \}`)
	customProperty = regexp.MustCompile(`--([a-z0-9-]+):\s*([^;]*\S)\s*$`)
	hexColour      = regexp.MustCompile(`^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$`)
	spacingToken   = regexp.MustCompile(`^s[0-9]+$`)
)

// ParseDesign derives the design document from the stylesheet: the dark
// scheme is the root declaration, the light scheme overrides it under
// prefers-color-scheme: light.
func ParseDesign(stylesheet string) (DesignDocument, error) {
	dark := rootBlock.FindStringSubmatch(stylesheet)
	light := lightRootBlock.FindStringSubmatch(stylesheet)
	if dark == nil || light == nil {
		return DesignDocument{}, errors.New("webui design: the stylesheet declares no root token block for each scheme")
	}
	base := properties(dark[1])
	document := DesignDocument{Source: StylesheetPath, Spacing: map[string]string{}, Type: map[string]string{}, Radius: map[string]string{}}
	for name, value := range base {
		switch {
		case spacingToken.MatchString(name):
			document.Spacing[name] = value
		case strings.HasPrefix(name, "t-"):
			document.Type[name] = value
		case strings.HasPrefix(name, "radius-"):
			document.Radius[name] = value
		}
	}
	overrides := properties(light[1])
	painted := paintedPairs(stylesheet)
	for _, scheme := range []struct {
		name   string
		tokens map[string]string
	}{{"dark", base}, {"light", merged(base, overrides)}} {
		measured, err := measureScheme(scheme.name, scheme.tokens, painted)
		if err != nil {
			return DesignDocument{}, err
		}
		document.Schemes = append(document.Schemes, measured)
	}
	return document, nil
}

// paintedPairs answers each token text colour the stylesheet paints on a
// token background within one rule, in the order the rules declare them.
// Rules are split at their braces and declarations at semicolons, so a
// pair is read as the stylesheet states it; comments hold no declarations.
func paintedPairs(stylesheet string) [][2]string {
	token := func(value string) string {
		name, ok := strings.CutPrefix(strings.TrimSpace(value), "var(--")
		if name, closed := strings.CutSuffix(name, ")"); ok && closed {
			return name
		}
		return ""
	}
	var uncommented strings.Builder
	for rest := stylesheet; rest != ""; {
		before, comment, opened := strings.Cut(rest, "/*")
		uncommented.WriteString(before)
		_, rest, _ = strings.Cut(comment, "*/")
		if !opened {
			break
		}
	}
	var pairs [][2]string
	for block := range strings.SplitSeq(uncommented.String(), "}") {
		_, body, ok := strings.Cut(block, "{")
		if !ok {
			continue
		}
		var foreground, background string
		for declaration := range strings.SplitSeq(body, ";") {
			property, value, ok := strings.Cut(declaration, ":")
			if !ok {
				continue
			}
			switch strings.TrimSpace(property) {
			case "color":
				foreground = token(value)
			case "background", "background-color":
				background = token(value)
			}
		}
		if pair := [2]string{foreground, background}; foreground != "" && background != "" && !slices.Contains(pairs, pair) {
			pairs = append(pairs, pair)
		}
	}
	return pairs
}

// EncodeDesign renders the document as it is published: indented JSON
// with a trailing newline, keys in a stable order.
func EncodeDesign(document DesignDocument) ([]byte, error) {
	encoded, err := json.MarshalIndent(document, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(encoded, '\n'), nil
}

func properties(block string) map[string]string {
	values := map[string]string{}
	for declaration := range strings.SplitSeq(block, ";") {
		if match := customProperty.FindStringSubmatch(declaration); match != nil {
			values[match[1]] = match[2]
		}
	}
	return values
}

func merged(base, overrides map[string]string) map[string]string {
	values := maps.Clone(base)
	maps.Copy(values, overrides)
	return values
}

func measureScheme(name string, tokens map[string]string, painted [][2]string) (DesignScheme, error) {
	scheme := DesignScheme{Name: name, Colours: map[string]string{}}
	colours := map[string]colour{}
	for token, value := range tokens {
		if parsed, ok := parseColour(value); ok {
			colours[token], scheme.Colours[token] = parsed, strings.ToLower(strings.TrimSpace(value))
		}
	}
	pairs := make([][2]string, 0, len(textTokens)*len(surfaceTokens)+len(painted))
	for _, foreground := range textTokens {
		for _, background := range surfaceTokens {
			pairs = append(pairs, [2]string{foreground, background})
		}
	}
	for _, pair := range painted {
		if !slices.Contains(pairs, pair) {
			pairs = append(pairs, pair)
		}
	}
	for _, pair := range pairs {
		foreground, okForeground := colours[pair[0]]
		background, okBackground := colours[pair[1]]
		if !okForeground || !okBackground {
			return DesignScheme{}, fmt.Errorf("webui design: %s %s on %s: a token is not a colour", name, pair[0], pair[1])
		}
		ratio := contrast(foreground, background)
		// A translucent background is read over whichever surface lies under
		// it, so its pair counts at the worst of them.
		if background.alpha < 1 {
			ratio = math.Inf(1)
			for _, surface := range surfaceTokens {
				ratio = min(ratio, contrast(foreground, background.over(colours[surface])))
			}
		}
		scheme.Contrast = append(scheme.Contrast, TokenContrast{Foreground: pair[0], Background: pair[1], Ratio: ratio})
	}
	slices.SortFunc(scheme.Contrast, func(a, b TokenContrast) int {
		return strings.Compare(a.Foreground+"/"+a.Background, b.Foreground+"/"+b.Background)
	})
	return scheme, nil
}

// colour is one token colour: its channels on the 0-255 scale and its
// alpha, which is whole for an opaque colour.
type colour struct {
	channels [3]float64
	alpha    float64
}

// parseColour reads a hex colour or an rgb()/rgba() colour; any other
// value (a shadow, a length) names no colour.
func parseColour(value string) (colour, bool) {
	value = strings.ToLower(strings.TrimSpace(value))
	parsed := colour{alpha: 1}
	if hexColour.MatchString(value) {
		digits := value[1:]
		if len(digits) == len(parsed.channels) {
			digits = string([]byte{digits[0], digits[0], digits[1], digits[1], digits[2], digits[2]})
		}
		for index := range parsed.channels {
			channel, err := strconv.ParseUint(digits[index*2:index*2+2], 16, 8)
			if err != nil {
				return colour{}, false
			}
			parsed.channels[index] = float64(channel)
		}
		return parsed, true
	}
	inner, functional := strings.CutPrefix(value, "rgba(")
	if !functional {
		inner, functional = strings.CutPrefix(value, "rgb(")
	}
	inner, closed := strings.CutSuffix(inner, ")")
	parts := strings.Split(inner, ",")
	if !functional || !closed || len(parts) < len(parsed.channels) || len(parts) > len(parsed.channels)+1 {
		return colour{}, false
	}
	for index, part := range parts {
		number, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
		if err != nil {
			return colour{}, false
		}
		if index < len(parsed.channels) {
			parsed.channels[index] = number
		} else {
			parsed.alpha = number
		}
	}
	return parsed, true
}

// over is the colour seen where a translucent colour lies on an opaque one.
func (c colour) over(ground colour) colour {
	seen := colour{alpha: ground.alpha}
	for index := range seen.channels {
		seen.channels[index] = c.channels[index]*c.alpha + ground.channels[index]*(1-c.alpha)
	}
	return seen
}

// contrastRatio is the WCAG ratio of two colour values, rounded to
// hundredths so the derived document is stable.
func contrastRatio(foreground, background string) (float64, error) {
	a, ok := parseColour(foreground)
	b, okBackground := parseColour(background)
	if !ok || !okBackground {
		return 0, fmt.Errorf("%q or %q is not a colour", foreground, background)
	}
	return contrast(a, b), nil
}

func contrast(foreground, background colour) float64 {
	a, b := relativeLuminance(foreground), relativeLuminance(background)
	return math.Round((max(a, b)+0.05)/(min(a, b)+0.05)*100) / 100
}

func relativeLuminance(c colour) float64 {
	var luminance float64
	for index, weight := range []float64{0.2126, 0.7152, 0.0722} {
		channel := c.channels[index] / 255
		if channel <= 0.03928 {
			channel /= 12.92
		} else {
			channel = math.Pow((channel+0.055)/1.055, 2.4)
		}
		luminance += weight * channel
	}
	return luminance
}
