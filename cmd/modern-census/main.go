// modern-census owns Overgo's measured conformance to its pinned modern-Go
// catalog. The catalog mode is the first campaign stage; later stages add the
// type-aware source census and monotonic baseline without changing authority.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/repoanalysis"
)

func main() {
	clioptions.MainNamed("modern-census", runCommand)
}

func runCommand() error { return run(os.Args[1:], os.Stdout) }

func run(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("modern-census", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	target := flags.String("go-version", repoanalysis.ModernGoTargetVersion, "Go major.minor version to evaluate")
	prior := flags.Bool("prior", false, "print the audited disposition of the prior modernization lane")
	censusMode := flags.Bool("census", false, "measure the repository source against every applicable guideline")
	checkMode := flags.Bool("check", false, "require complete measured coverage within the permanent baseline")
	workMode := flags.Bool("work", false, "select bounded work and derive its package verification")
	writeBaseline := flags.Bool("write-baseline", false, "write the current census to "+repoanalysis.ModernGoBaselineFile)
	lowerBaseline := flags.Bool("lower-baseline", false, "refresh the baseline while only lowering guideline ceilings")
	fixTestingContext := flags.Bool("fix-testing-context", false, "bind type-proven test contexts to their testing owner")
	fixBenchmarkLoop := flags.Bool("fix-benchmark-loop", false, "replace index-free testing.B.N ranges with testing.B.Loop")
	fixErrorIdentity := flags.Bool("fix-error-identity", false, "replace type-proven direct error comparisons with errors.Is")
	fixJSONOmitZero := flags.Bool("fix-json-omitzero", false, "replace wire-equivalent scalar omitempty tags with omitzero")
	fixExtrema := flags.Bool("fix-extrema", false, "replace type-proven extrema with min, max, slices.Min, and slices.Max")
	fixNumericRange := flags.Bool("fix-numeric-range", false, "replace type-proven zero-based numerical loops with range-over-integer")
	fixManualIdioms := flags.Bool("fix-manual-idioms", false, "rewrite proven fallback, ticker, typed-sort, and slice-clone idioms")
	closeExceptions := flags.Bool("close-exceptions", false, "replace every retained candidate with exact owned exception authority")
	exceptionExpiry := flags.String("expires", "", "required YYYY-MM-DD expiry for -close-exceptions")
	publishCensus := flags.Bool("publish-census", false, "write source-bound resolved-guideline closure evidence")
	rules := flags.String("rules", "", "comma-separated plan-owned guideline IDs for -work")
	limit := flags.String("limit", "", "required maximum source sites for -work")
	root := flags.String("root", ".", "repository root for -census or -work")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return fmt.Errorf("unexpected arguments: %v", flags.Args())
	}
	selectedModes := 0
	for _, selected := range []bool{*censusMode, *checkMode, *workMode, *writeBaseline, *lowerBaseline, *fixTestingContext, *fixBenchmarkLoop, *fixErrorIdentity, *fixJSONOmitZero, *fixExtrema, *fixNumericRange, *fixManualIdioms, *closeExceptions, *publishCensus} {
		if selected {
			selectedModes++
		}
	}
	if selectedModes > 1 {
		return fmt.Errorf("modern-census modes are mutually exclusive")
	}
	if !*closeExceptions && *exceptionExpiry != "" {
		return fmt.Errorf("-expires requires -close-exceptions")
	}
	if *closeExceptions {
		return closeModernGoExceptions(*root, *target, *exceptionExpiry, output)
	}
	if *publishCensus {
		return publishModernGoCensus(*root, *target, output)
	}
	if *fixNumericRange {
		count, files, err := repoanalysis.RewriteModernGoNumericIntegerRanges(*root)
		if err != nil {
			return err
		}
		census, err := repoanalysis.BuildModernGoCensus(*root, *target)
		if err != nil {
			return err
		}
		retained := 0
		for _, finding := range census.Findings {
			if finding.ID != "range_over_int" {
				continue
			}
			for _, site := range finding.Candidates {
				if site.NumericRuntime {
					retained++
				}
			}
		}
		fmt.Fprintf(output, "modern-census: numeric-range rewritten=%d files=%d retained=%d source=typed-stable-integer-bounds\n", count, len(files), retained)
		return nil
	}
	if *fixManualIdioms {
		result, err := repoanalysis.RewriteModernGoManualIdioms(*root)
		if err != nil {
			return err
		}
		census, err := repoanalysis.BuildModernGoCensus(*root, *target)
		if err != nil {
			return err
		}
		retained := 0
		for _, finding := range census.Findings {
			switch finding.ID {
			case "cmp_or", "time_tick_gc", "slices_clone", "slices_sort":
				retained += len(finding.Candidates)
			}
		}
		fmt.Fprintf(output,
			"modern-census: manual-idioms fallback=%d ticker=%d slice-clone=%d typed-sort=%d files=%d retained=%d source=typed-eager-safe-equivalence\n",
			result.Fallbacks, result.Tickers, result.SliceClones, result.TypedSorts, len(result.Files), retained)
		return nil
	}
	if *fixExtrema {
		count, files, err := repoanalysis.RewriteModernGoExtrema(*root)
		if err != nil {
			return err
		}
		census, err := repoanalysis.BuildModernGoCensus(*root, *target)
		if err != nil {
			return err
		}
		retained := 0
		for _, finding := range census.Findings {
			if finding.ID == "min_max" || finding.ID == "slices_max_min" {
				retained += len(finding.Candidates)
			}
		}
		fmt.Fprintf(output, "modern-census: extrema rewritten=%d files=%d retained=%d source=typed-ordered-extrema-policy\n", count, len(files), retained)
		return nil
	}
	if *fixErrorIdentity {
		count, files, err := repoanalysis.RewriteModernGoErrorIdentityComparisons(*root)
		if err != nil {
			return err
		}
		fmt.Fprintf(output, "modern-census: error-identity rewritten=%d files=%d source=typed-error-comparison\n", count, len(files))
		return nil
	}
	if *fixJSONOmitZero {
		count, files, err := repoanalysis.RewriteModernGoOmitZeroEquivalent(*root)
		if err != nil {
			return err
		}
		census, err := repoanalysis.BuildModernGoCensus(*root, *target)
		if err != nil {
			return err
		}
		retained := 0
		for _, finding := range census.Findings {
			if finding.ID == "json_omitzero" {
				retained = len(finding.Candidates)
				break
			}
		}
		fmt.Fprintf(output, "modern-census: json-omitzero rewritten=%d files=%d retained=%d source=typed-scalar-equivalence\n", count, len(files), retained)
		return nil
	}
	if *fixBenchmarkLoop {
		count, files, err := repoanalysis.RewriteModernGoBenchmarkLoops(*root)
		if err != nil {
			return err
		}
		fmt.Fprintf(output, "modern-census: benchmark-loop rewritten=%d files=%d source=testing-b-loop\n", count, len(files))
		return nil
	}
	if *fixTestingContext {
		count, files, err := repoanalysis.RewriteModernGoTestingContexts(*root)
		if err != nil {
			return err
		}
		fmt.Fprintf(output, "modern-census: testing-context rewritten=%d files=%d source=typed-testing-owner\n", count, len(files))
		return nil
	}
	if *lowerBaseline {
		return lowerModernGoBaseline(*root, *target, output)
	}
	if *checkMode {
		return checkModernGoBaseline(*root, *target, output)
	}
	if *writeBaseline {
		return writeModernGoBaseline(*root, *target, output)
	}
	if *workMode {
		return printWork(*root, *target, *rules, *limit, output)
	}
	if *censusMode {
		return printCensus(*root, *target, output)
	}
	catalog := repoanalysis.ModernGoCatalog()
	applicable, err := repoanalysis.ModernGoApplicableGuidelines(*target)
	if err != nil {
		return err
	}
	for _, guideline := range applicable {
		fmt.Fprintf(output, "%s go%s: %s\n", guideline.ID, guideline.SinceVersion, guideline.Guideline)
	}
	if *prior {
		for _, commit := range repoanalysis.ModernGoPriorCampaign() {
			fmt.Fprintf(output, "prior %s %s: %s -- %s\n",
				commit.Commit, commit.Disposition, commit.Subject, commit.Rationale)
		}
	}
	fmt.Fprintf(output,
		"modern-census: catalog=%d applicable=%d excluded=%d target=go%s repository=%s source=%s plugin=%s sha256=%s\n",
		len(catalog), len(applicable), len(catalog)-len(applicable), *target,
		repoanalysis.ModernGoCatalogRepository,
		repoanalysis.ModernGoCatalogCommit, repoanalysis.ModernGoCatalogPluginVersion,
		repoanalysis.ModernGoCatalogSHA256,
	)
	return nil
}

func checkModernGoBaseline(root, target string, output io.Writer) error {
	census, err := repoanalysis.BuildModernGoCensus(root, target)
	if err != nil {
		return err
	}
	name := filepath.Join(root, filepath.FromSlash(repoanalysis.ModernGoBaselineFile))
	baseline, err := repoanalysis.LoadModernGoBaseline(name)
	if err != nil {
		return err
	}
	if err := repoanalysis.AdmitModernGoRatchet(baseline, census, time.Now().UTC()); err != nil {
		return err
	}
	expected, err := repoanalysis.BuildModernGoPublishedCensus(census, baseline)
	if err != nil {
		return err
	}
	publishedName := filepath.Join(root, filepath.FromSlash(repoanalysis.ModernGoPublishedCensusFile))
	var published repoanalysis.ModernGoPublishedCensus
	if err := jsonfile.DecodeStrict(publishedName, &published); err != nil {
		return err
	}
	if err := repoanalysis.ValidateModernGoPublishedCensus(published, expected); err != nil {
		return err
	}
	measured := 0
	for _, finding := range census.Findings {
		if finding.Measured {
			measured++
		}
	}
	fmt.Fprintf(output,
		"modern-census: check=pass measured=%d/%d candidates=%d exceptions=%d unresolved=%d excluded=%d authority=%s baseline=%s published=%s source=%s catalog=%s\n",
		measured, len(census.Findings), census.CandidateCount(), len(baseline.Exceptions), published.Unresolved, len(published.Excluded),
		baseline.ExceptionSHA256, repoanalysis.ModernGoBaselineFile, repoanalysis.ModernGoPublishedCensusFile,
		census.SourceIdentity, census.CatalogCommit)
	return nil
}

func publishModernGoCensus(root, target string, output io.Writer) error {
	census, err := repoanalysis.BuildModernGoCensus(root, target)
	if err != nil {
		return err
	}
	baselineName := filepath.Join(root, filepath.FromSlash(repoanalysis.ModernGoBaselineFile))
	baseline, err := repoanalysis.LoadModernGoBaseline(baselineName)
	if err != nil {
		return err
	}
	if err := repoanalysis.AdmitModernGoRatchet(baseline, census, time.Now().UTC()); err != nil {
		return err
	}
	published, err := repoanalysis.BuildModernGoPublishedCensus(census, baseline)
	if err != nil {
		return err
	}
	name := filepath.Join(root, filepath.FromSlash(repoanalysis.ModernGoPublishedCensusFile))
	if err := jsonfile.Write(name, published, clioptions.OutputFileMode); err != nil {
		return err
	}
	fmt.Fprintf(output,
		"modern-census: published=%s applicable=%d measured=%d adopted=%d candidates=%d excepted=%d unresolved=%d excluded=%d authority=%s source=%s\n",
		repoanalysis.ModernGoPublishedCensusFile, published.Applicable, published.Measured, published.Adopted,
		published.Candidates, published.ExceptedCandidates, published.Unresolved, len(published.Excluded),
		published.ExceptionSHA256, published.SourceIdentity)
	return nil
}

func closeModernGoExceptions(root, target, expires string, output io.Writer) error {
	if expires == "" {
		return fmt.Errorf("-close-exceptions requires -expires")
	}
	census, err := repoanalysis.BuildModernGoCensus(root, target)
	if err != nil {
		return err
	}
	name := filepath.Join(root, filepath.FromSlash(repoanalysis.ModernGoBaselineFile))
	baseline, err := repoanalysis.LoadModernGoBaseline(name)
	if err != nil {
		return err
	}
	closed, err := repoanalysis.CloseModernGoExceptions(baseline, census, expires, time.Now().UTC())
	if err != nil {
		return err
	}
	if err := jsonfile.Write(name, closed, clioptions.OutputFileMode); err != nil {
		return err
	}
	fmt.Fprintf(output,
		"modern-census: exceptions=closed groups=%d candidates=%d expires=%s authority=%s baseline=%s\n",
		len(closed.Exceptions), census.CandidateCount(), expires, closed.ExceptionSHA256, repoanalysis.ModernGoBaselineFile)
	return nil
}

func writeModernGoBaseline(root, target string, output io.Writer) error {
	census, err := repoanalysis.BuildModernGoCensus(root, target)
	if err != nil {
		return err
	}
	name := filepath.Join(root, filepath.FromSlash(repoanalysis.ModernGoBaselineFile))
	baseline, err := repoanalysis.BuildModernGoBaseline(census)
	if err != nil {
		return err
	}
	if err := jsonfile.Write(name, baseline, clioptions.OutputFileMode); err != nil {
		return err
	}
	fmt.Fprintf(output, "modern-census: baseline=%s findings=%d source=%s catalog=%s\n",
		repoanalysis.ModernGoBaselineFile, len(census.Findings), census.SourceIdentity, census.CatalogCommit)
	return nil
}

func lowerModernGoBaseline(root, target string, output io.Writer) error {
	census, err := repoanalysis.BuildModernGoCensus(root, target)
	if err != nil {
		return err
	}
	name := filepath.Join(root, filepath.FromSlash(repoanalysis.ModernGoBaselineFile))
	baseline, err := repoanalysis.LoadModernGoBaseline(name)
	if err != nil {
		return err
	}
	lowered, err := repoanalysis.LowerModernGoBaseline(baseline, census)
	if err != nil {
		return err
	}
	if err := jsonfile.Write(name, lowered, clioptions.OutputFileMode); err != nil {
		return err
	}
	fmt.Fprintf(output, "modern-census: lowered baseline=%s candidates=%d source=%s catalog=%s\n",
		repoanalysis.ModernGoBaselineFile, census.CandidateCount(), census.SourceIdentity, census.CatalogCommit)
	return nil
}

func printWork(root, target, rules, limitText string, output io.Writer) error {
	limit, err := strconv.Atoi(limitText)
	if err != nil {
		return fmt.Errorf("parse -limit: %w", err)
	}
	selection, err := repoanalysis.BuildModernGoWorkSelection(root, target, repoanalysis.ModernGoWorkRequest{
		Guidelines: strings.FieldsFunc(rules, func(value rune) bool { return value == ',' }),
		Limit:      limit,
	})
	if err != nil {
		return err
	}
	for _, finding := range selection.Findings {
		fmt.Fprintf(output, "%s risk=%s selected=%d\n", finding.ID, finding.Risk, len(finding.Sites))
		for _, site := range finding.Sites {
			fmt.Fprintf(output, "site %s:%d package=%s symbol=%s test=%s generated=%s typed=%s numeric=%s\n",
				site.Path, site.Line, site.Package, site.Symbol,
				strconv.FormatBool(site.Test), strconv.FormatBool(site.Generated), strconv.FormatBool(site.TypeChecked),
				strconv.FormatBool(site.NumericRuntime))
		}
	}
	if len(selection.Verification.DirectArgs) != 0 {
		fmt.Fprintf(output, "verify-direct go %s\n", strings.Join(selection.Verification.DirectArgs, " "))
		fmt.Fprintf(output, "verify-closure go %s\n", strings.Join(selection.Verification.ClosureArgs, " "))
	}
	fmt.Fprintf(output,
		"modern-work: inspected=%d matched=%d selected=%d excluded=%d unmeasured=%d target=go%s build=%s source=%s catalog=%s\n",
		selection.Coverage.Inspected, selection.Coverage.Matched, selection.Coverage.Selected,
		selection.Coverage.Excluded, selection.Coverage.Unmeasured, selection.TargetGo,
		selection.BuildContext, selection.SourceIdentity, selection.CatalogCommit)
	return nil
}

func printCensus(root, target string, output io.Writer) error {
	census, err := repoanalysis.BuildModernGoCensus(root, target)
	if err != nil {
		return err
	}
	for _, finding := range census.Findings {
		fmt.Fprintf(output, "%s risk=%s inspected=%d typed=%d candidates=%d adopted=%d\n",
			finding.ID, finding.Risk, finding.InspectedFiles, finding.TypedFiles,
			len(finding.Candidates), len(finding.Adopted))
	}
	fmt.Fprintf(output,
		"modern-census: findings=%d candidates=%d adopted=%d target=go%s build=%s source=%s catalog=%s\n",
		len(census.Findings), census.CandidateCount(), census.AdoptedCount(), census.TargetGo,
		census.BuildContext, census.SourceIdentity, census.CatalogCommit)
	return nil
}
