import json
import os

from simulator import save_report, save_json, simulate
from agents import skeleton_of, node_count, combine_with_justification, failed_criteria, template_family_balance
from selection import TrialRegistry, promotion_eligible, DSR_PROMOTE, set_scope
from calibration import expected_real_adjustment

COMPETITION_SCOPE = "competition"

LESSONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'reports', 'lessons.jsonl')


def load_lessons(path=LESSONS_FILE):
    """Dead-end fix records: {(failed-gates, fix-action): count}.

    Written by Competition from failed rounds, read by Optimizer to skip
    fixes that repeatedly dead-end on the same failure signature (>=3).
    Missing/corrupt file -> {} (safe default).
    """
    out = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if not e.get('round_passed'):
                    out[(tuple(e.get('failed', [])), e.get('fix', ''))] = \
                        out.get((tuple(e.get('failed', [])), e.get('fix', '')), 0) + 1
    except OSError:
        pass
    return out


def append_lessons(entries, path=LESSONS_FILE):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a') as f:
            for e in entries:
                f.write(json.dumps(e, default=str) + '\n')
    except OSError:
        pass

class Competition:
    def __init__(self, panel, refs, teams, rounds=3, max_opt_iter=5, outdir='reports',
                 settings=None):
        self.panel = panel
        self.refs = refs
        self.teams = teams
        self.rounds = rounds
        self.max_opt_iter = max_opt_iter
        self.outdir = outdir
        self.settings = settings
        os.makedirs(outdir, exist_ok=True)
        for t in teams:
            t['history'] = []
            t['wins'] = 0
            t['passed_count'] = 0
            t['best_fitness'] = None
            t['skeletons'] = set()  # Phase A1: final-expression skeletons, cross-round dup detection
            t['rescue'] = []  # Phase B5: near-miss finals (failed exactly 1 gate, no error)
        fams = template_family_balance([tpl for t in teams for tpl in t['ideator'].templates])
        print(f"[Competition] template families: {fams['counts']} ({fams['balancer']})")
        # Scoped registry: this competition's trials (including optimizer
        # internals) count toward ITS multiplicity only, never GP campaigns'.
        set_scope(COMPETITION_SCOPE)

    def _own_refs(self, team):
        return [{'team': team['id'], 'name': f"round{h['round']}", 'pnl': h['pnl'], 'sharpe': h['sharpe']}
                for h in team['history']]

    def run_round(self, k):
        print()
        print(f"================ ROUND {k} ================")
        results = []
        # RR-14/Romano-Wolf: StepM peers = top-3 refs by Sharpe, fixed for the
        # round so every final faces identical multiplicity adjustment.
        _peers = [r["pnl"] for r in sorted(self.refs, key=lambda r: r["sharpe"],
                                           reverse=True)[:3]]
        # Expected-real lens: per-final family bias envelope (falls back to
        # the global table when a family has <3 pairs; self-disables when
        # even the global pool is thin). Computed per final below.
        from calibration import family_of_expression as _fam
        for team in self.teams:
            print(f"\n--- {team['name']} ---")
            idea = team['ideator'].propose(k)
            # Phase A1 pre-simulate filter: skip ideator proposals whose
            # skeleton duplicates a prior FINAL from the same team (param-only
            # resubmission across rounds). Bounded by template pool size.
            skips = 0
            while (skeleton_of(idea.expr) in team['skeletons']
                   and skips < len(team['ideator'].templates)):
                print(f"[Ideator] skeleton duplicate of a prior final — requesting next template "
                      f"({idea.expr[:60]}...)")
                idea = team['ideator'].propose(k)
                skips += 1
            print(f"[{team['ideator'].name} Ideator] initial expression ({idea.name}):")
            print(f"    {idea.expr}")
            print(f"    rationale: {idea.rationale}")

            expr, rep, olog = team['optimizer'].optimize(idea, self.panel, self.refs,
                                                         self._own_refs(team), self.max_opt_iter,
                                                         self.settings)
            for e in olog:
                if 'rejected_discouraged_pattern' in e:
                    print(f"[{team['optimizer'].name} Optimizer] iter {e['iteration']}: {e['rejected_discouraged_pattern']}")
                    continue
                if 'rejected_skeleton_duplicate' in e:
                    print(f"[{team['optimizer'].name} Optimizer] iter {e['iteration']}: skeleton duplicate, skipping sim ({e['rejected_skeleton_duplicate']})")
                    continue
                if 'rejected_dimension_violation' in e:
                    print(f"[{team['optimizer'].name} Optimizer] iter {e['iteration']}: dimension violation, skipping sim ({e['rejected_dimension_violation']})")
                    continue
                if 'rejected_lesson' in e:
                    print(f"[{team['optimizer'].name} Optimizer] iter {e['iteration']}: lesson skip ({e['rejected_lesson']})")
                    continue
                if 'diagnosis' in e:
                    print(f"[{team['optimizer'].name} Optimizer] iter {e['iteration']}:")
                    print(f"    report    : {e['metrics']}  FAILED: {', '.join(e['failed'])}")
                    print(f"    diagnosis : {e['diagnosis']}")
                    print(f"    hypothesis: {e['hypothesis']}")
                    print(f"    fix       : {e['fix']}")
                    print(f"    new expr  : {e['new_expr']}")
                else:
                    print(f"[{team['optimizer'].name} Optimizer] iter {e['iteration']}: {e['action']}")
                    if 'expr' in e:
                        print(f"    report    : {e['metrics']}  FAILED: {', '.join(e['failed'])}")

            passed = bool(rep.get('passed'))
            # Core Operational Invariant 2 — promotion block: gates-passed is
            # necessary but not sufficient. Promotion additionally requires
            # lineage (dataset_id + logged N_trials) and DSR >= threshold.
            # Unlogged trials get DSR = 0 and are BLOCKED from history/
            # leaderboard even if all six gates pass.
            _bias = expected_real_adjustment(_fam(rep.get("expression", "")))
            eligible, promo = promotion_eligible(rep, TrialRegistry(), DSR_PROMOTE,
                                                   COMPETITION_SCOPE,
                                                   peer_pnls=_peers, expect_real=_bias)
            if passed and not eligible:
                print(f"[Promotion] BLOCKED: {'; '.join(promo['reasons'])}")
            elif passed:
                print(f"[Promotion] ELIGIBLE: DSR={promo['dsr']} N={promo['n_trials']} ds={promo['dataset_id']}")
            if passed and promo.get("expected_real") is not None:
                er = promo["expected_real"]
                print(f"[Expected-real:{promo.get('expected_real_family', '?')}] adj Sharpe {er.get('sharpe')} / Fitness {er.get('fitness')} "
                      f"(bias-adjusted; gates evaluated on sim metrics above)")
            # Phase B5: elite crossover — if the final failed but the team has
            # a passing history, try ONE scheduled cross (best history x
            # current final, add-op). Adopted only if it passes AND beats the
            # current final on fitness. Logged either way; never silent.
            if not passed and not rep.get('error') and team['history']:
                best_h = max(team['history'], key=lambda h: h['fitness'])
                combo, why = combine_with_justification(
                    [best_h['expr'], expr], '+',
                    "elite crossover: best prior passer x current final; "
                    "add preserves either-leg contribution")
                crep = simulate(combo, self.panel, self.refs,
                                self._own_refs(team), settings=self.settings)
                if crep.get('error'):
                    print(f"[Crossover] ERROR: {crep['error']}")
                else:
                    cm = crep['metrics']
                    print(f"[Crossover] {combo[:100]}")
                    print(f"    sharpe={cm['sharpe']:.2f} fitness={cm['fitness']:.2f} "
                          f"passed={bool(crep.get('passed'))} "
                          f"failed={failed_criteria(crep)} :: {why[:80]}...")
                    if bool(crep.get('passed')) and cm['fitness'] > rep['metrics']['fitness']:
                        _cbias = expected_real_adjustment(_fam(combo))
                        celig, cpromo = promotion_eligible(crep, TrialRegistry(), DSR_PROMOTE,
                                                             COMPETITION_SCOPE,
                                                             peer_pnls=_peers, expect_real=_cbias)
                        if celig:
                            print("    ADOPTED as team submission (passed and beats final)")
                            expr, rep, passed = combo, crep, True
                            eligible, promo = celig, cpromo
                        else:
                            print(f"    crossover passes gates but BLOCKED: {'; '.join(cpromo['reasons'])}")
            print(f"FINAL expression ({'PASS' if passed else 'FAIL'}):")
            print(f"    {expr}")
            self._print_report(rep)
            results.append((team, expr, rep, passed))
            sk = skeleton_of(expr)
            if sk is not None:
                team['skeletons'].add(sk)

            if passed and eligible:
                team['passed_count'] += 1
                fit = rep['metrics']['fitness']
                if team['best_fitness'] is None or fit > team['best_fitness']:
                    team['best_fitness'] = fit
                team['history'].append({'round': k, 'expr': expr, 'pnl': rep['pnl'],
                                        'sharpe': rep['metrics']['sharpe'], 'fitness': fit})
            elif passed:
                print("    !! passed gates but not promoted (lineage/DSR block above)")
            elif rep.get('error'):
                print(f"    !! simulator error: {rep['error']}")
            else:
                # Phase B5 rescue pool: failed exactly 1 gate -> scheduled retry
                # material for future rounds (kept best 3 by fitness). Phase B6:
                # log dead-end (failed-gates, fix) lessons for the Optimizer.
                failed = failed_criteria(rep)
                if len(failed) == 1:
                    team['rescue'].append({'round': k, 'expr': expr, 'failed': failed,
                                           'fitness': rep['metrics']['fitness']})
                    team['rescue'] = sorted(team['rescue'],
                                            key=lambda r: r['fitness'], reverse=True)[:3]
                    print(f"[Rescue] pooled near-miss (failed only {failed})")
            lessons = [{'round': k, 'team': team['id'],
                        'skeleton': skeleton_of(expr),
                        'failed': sorted(failed_criteria(rep)),
                        'fix': e.get('fix', ''), 'round_passed': passed}
                       for e in olog if 'fix' in e]
            append_lessons(lessons)

        round_json = {'round': k, 'teams': []}
        winner = None
        # Invariant 2: round winner must be promotion-eligible (gates + lineage
        # + DSR), not merely gates-passing.
        passing = []
        for t, e, r, p in results:
            if not p:
                continue
            _elig, _ = promotion_eligible(r, TrialRegistry(), DSR_PROMOTE,
                                              COMPETITION_SCOPE, peer_pnls=_peers,
                                              expect_real=expected_real_adjustment(_fam(e)))
            if _elig:
                passing.append((t, e, r))
        if passing:
            passing.sort(key=lambda x: (x[2]['metrics']['fitness'], x[2]['metrics']['sharpe']), reverse=True)
            winner = passing[0]
            for t, _, _, _ in results:
                if t is winner[0]:
                    t['wins'] += 1
        for team, expr, rep, passed in results:
            slim = {kk: vv for kk, vv in rep.items() if kk != 'pnl'}
            # Phase B8: AlphaEval-style info metrics (display only, never scored).
            ys = [y['sharpe'] for y in rep.get('yearly', [])]
            import numpy as _np
            stability = round(float(1.0 / (1.0 + _np.std(ys))), 4) if ys else None
            complexity = node_count(expr)
            _elig, _promo = promotion_eligible(rep, TrialRegistry(), DSR_PROMOTE,
                                                 COMPETITION_SCOPE, peer_pnls=_peers,
                                                 expect_real=expected_real_adjustment(_fam(expr)))
            round_json['teams'].append({'team': team['name'], 'final_expression': expr,
                                        'passed': passed, 'report': slim,
                                        'stability': stability, 'complexity': complexity,
                                        'promotion': {'eligible': _elig, 'info': _promo},
                                        'rescue_pool': team['rescue']})
            print(f"[{team['name']}] stability={stability} complexity={complexity} "
                  f"rescue={[r['round'] for r in team['rescue']]}")
        round_json['winner'] = winner[0]['name'] if winner else None
        save_json(round_json, os.path.join(self.outdir, f'round_{k}.json'))

        print(f"\n--- ROUND {k} WINNER ---")
        if winner:
            team, expr, rep = winner
            others = [(t, r) for t, e, r, p in results if t is not team]
            print(f"{team['name']} — fitness {rep['metrics']['fitness']:.2f}, Sharpe {rep['metrics']['sharpe']:.2f}")
            for t, r in others:
                if r.get('passed'):
                    print(f"  (beat {t['name']}: fitness {r['metrics']['fitness']:.2f}, Sharpe {r['metrics']['sharpe']:.2f})")
                elif r.get('error'):
                    print(f"  ({t['name']} failed: simulator error)")
                else:
                    fails = [kk for kk, vv in r['criteria'].items() if not vv['pass']]
                    print(f"  ({t['name']} failed cutoffs: {', '.join(fails)})")
        else:
            if any(p for _, _, _, p in results):
                print("No promotable alpha — gates passed but promotion blocked (lineage/DSR).")
            else:
                print("No pass — both teams failed all hard cutoffs this round.")
        self._print_leaderboard()

    def _print_report(self, rep):
        if rep.get('error'):
            return
        m = rep['metrics']
        print("    metrics:")
        print(f"      Sharpe {m['sharpe']:.2f} | Fitness {m['fitness']:.2f} | Returns {m['returns_pct']:.1f}% | "
              f"Turnover {m['turnover_pct']:.1f}% | Drawdown {m['drawdown_pct']:.1f}% | "
              f"Margin {m.get('margin_bps', 0.0):.1f}bps | Net {m.get('returns_net_pct', 0.0):.1f}% (cost {m.get('cost_drag_pct', 0.0):.1f}%)")
        print(f"      Max weight {m['weight_concentration_pct']:.1f}% on {rep['max_weight_date']} | "
              f"Sub-universe Sharpe {rep['criteria']['subuniverse_sharpe']['value']:.2f} "
              f"(cutoff {rep.get('subuniverse_cutoff')}) | "
              f"Self-corr {rep['criteria']['self_correlation']['value']:.2f}")
        if rep.get('risk'):
            rr = rep['risk']
            h95 = rr.get('hist', {}).get('0.95', {})
            t99 = rr.get('t', {}).get('0.99', {})
            ex = rr.get('exceedance', {})
            print(f"      Risk lens: VaR95 {h95.get('var', float('nan')):.4f} | ES95 {h95.get('es', float('nan')):.4f} | "
                  f"t-ES99 {t99.get('es', float('nan')):.4f} (gap {rr.get('es_gap_t_vs_normal_99', float('nan')):+.4f}) | "
                  f"cond t-ES {rr.get('cond_t_es_latest', float('nan')):.4f} | "
                  f"exceed {ex.get('exceed', '?')}/{ex.get('n', '?')} (exp {ex.get('expected', '?')}, {ex.get('verdict', '?')})")
        if rep.get('max_weight_dates'):
            tops = ', '.join(f"{w['date']} #{w['stock']} {w['weight_pct']:.2f}%"
                             for w in rep['max_weight_dates'])
            print(f"      Top weight dates: {tops}")
        if rep.get('self_corr_detail'):
            d = rep['self_corr_detail']
            bl = ', '.join(f"'{b['ref']}' Sh {b['ref_sharpe']:.2f} c {b['corr']:.2f} r {b['ratio']:.2f}{'OK' if b['pass'] else 'BLOCK'}"
                            for b in d.get('blockers', []))
            print(f"      Self-corr detail ({d.get('window', '?')}): candidate Sharpe {d.get('candidate_sharpe')} "
                  f"(need >= {d.get('required_ratio')}x each correlated ref): {bl}")
        elif rep.get('top_corr_ref'):
            print(f"      Top corr ref: '{rep['top_corr_ref']}' Sharpe {rep.get('top_corr_ref_sharpe')} "
                  f"signed {rep.get('top_corr_signed')} ({rep.get('top_corr_window_days')}d window)")
        print("    criteria:")
        for name, c in rep['criteria'].items():
            mark = 'PASS' if c['pass'] else 'FAIL'
            print(f"      [{mark}] {name}: {c['value']} (need {c['requirement']})")
        print("    yearly (counts warn when thin vs universe):")
        for y in rep['yearly']:
            thin = ' THIN-COVERAGE' if y.get('thin_coverage') else ''
            print(f"      {y['year']}: Sharpe {y['sharpe']:+.2f} | Fitness {y['fitness']:.2f} | "
                  f"Turnover {y['turnover_pct']:.1f}% | Returns {y['returns_pct']:+.1f}% | "
                  f"long {y.get('mean_long')}/{y.get('min_long')} short {y.get('mean_short')}/{y.get('min_short')}{thin}")
        if rep.get('periods'):
            per = ' | '.join(f"{k}: Sh {v['sharpe']:+.2f}/Fit {v['fitness']:.2f}" for k, v in rep['periods'].items())
            print(f"    periods: {per}")

    def _print_leaderboard(self):
        print("\n--- LEADERBOARD ---")
        rows = sorted(self.teams, key=lambda t: (-t['wins'], -(t['best_fitness'] or 0)))
        for t in rows:
            bf = f"{t['best_fitness']:.2f}" if t['best_fitness'] is not None else "n/a"
            print(f"  {t['name']}: wins {t['wins']} | passing alphas {t['passed_count']} | best fitness {bf}")
        save_json([{'team': t['name'], 'wins': t['wins'], 'passed': t['passed_count'],
                    'best_fitness': t['best_fitness'],
                    'submissions': [{'round': h['round'], 'expr': h['expr'],
                                     'sharpe': h['sharpe'], 'fitness': h['fitness']} for h in t['history']]}
                   for t in self.teams], os.path.join(self.outdir, 'leaderboard.json'))

    def run(self):
        for k in range(1, self.rounds + 1):
            self.run_round(k)
        print("\n================ COMPETITION COMPLETE ================")
        total_passing = sum(t['passed_count'] for t in self.teams)
        if total_passing == 0:
            print("No alphas passed all hard cutoffs — no overall winner (tie at 0 wins, 0 passing alphas).")
            print("The best-scoring failing candidates above illustrate the optimizer's search, not a winning submission.")
            champ = None
        else:
            champ = sorted(self.teams, key=lambda t: (-t['wins'], -(t['best_fitness'] or 0)))[0]
            print(f"Overall winner: {champ['name']} ({champ['wins']} round wins, "
                  f"best fitness {champ['best_fitness'] if champ['best_fitness'] is not None else 'n/a'})")
        self._record_run(champ)

    def _record_run(self, champ):
        """Phase 5 run manifest: one JSONL line per competition with code
        version, environment, config, and outcome summary (reports/runs.jsonl,
        gitignored). Reproduce from: commit + seed + rounds + settings."""
        import datetime
        import sys
        try:
            import numpy as _np
            np_ver = _np.__version__
        except ImportError:
            np_ver = "unknown"
        try:
            import subprocess as _sp
            sha = _sp.check_output(["git", "rev-parse", "--short", "HEAD"],
                                   cwd=os.path.dirname(os.path.abspath(__file__)),
                                   stderr=_sp.DEVNULL).decode().strip()
            dirty = bool(_sp.check_output(["git", "status", "--short"],
                                          cwd=os.path.dirname(os.path.abspath(__file__)),
                                          stderr=_sp.DEVNULL).decode().strip())
        except Exception:
            sha, dirty = "unknown", None
        entry = {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "git_sha": sha, "git_dirty": dirty,
                 "python": sys.version.split()[0], "numpy": np_ver,
                 "rounds": self.rounds, "max_opt_iter": self.max_opt_iter,
                 "settings": self.settings,
                 "teams": [{"name": t["name"], "wins": t["wins"],
                            "passed": t["passed_count"],
                            "best_fitness": t["best_fitness"]} for t in self.teams],
                 "champion": champ["name"] if champ is not None else None}
        try:
            with open(os.path.join(self.outdir, "runs.jsonl"), "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            pass
