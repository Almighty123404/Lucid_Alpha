import os

from simulator import save_report, save_json

class Competition:
    def __init__(self, panel, refs, teams, rounds=3, max_opt_iter=5, outdir='reports'):
        self.panel = panel
        self.refs = refs
        self.teams = teams
        self.rounds = rounds
        self.max_opt_iter = max_opt_iter
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        for t in teams:
            t['history'] = []
            t['wins'] = 0
            t['passed_count'] = 0
            t['best_fitness'] = None

    def _own_refs(self, team):
        return [{'team': team['id'], 'name': f"round{h['round']}", 'pnl': h['pnl'], 'sharpe': h['sharpe']}
                for h in team['history']]

    def run_round(self, k):
        print()
        print(f"================ ROUND {k} ================")
        results = []
        for team in self.teams:
            print(f"\n--- {team['name']} ---")
            idea = team['ideator'].propose(k)
            print(f"[{team['ideator'].name} Ideator] initial expression ({idea.name}):")
            print(f"    {idea.expr}")
            print(f"    rationale: {idea.rationale}")

            expr, rep, olog = team['optimizer'].optimize(idea, self.panel, self.refs,
                                                         self._own_refs(team), self.max_opt_iter)
            for e in olog:
                if 'rejected_discouraged_pattern' in e:
                    print(f"[{team['optimizer'].name} Optimizer] iter {e['iteration']}: {e['rejected_discouraged_pattern']}")
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
            print(f"FINAL expression ({'PASS' if passed else 'FAIL'}):")
            print(f"    {expr}")
            self._print_report(rep)
            results.append((team, expr, rep, passed))

            if passed:
                team['passed_count'] += 1
                fit = rep['metrics']['fitness']
                if team['best_fitness'] is None or fit > team['best_fitness']:
                    team['best_fitness'] = fit
                team['history'].append({'round': k, 'expr': expr, 'pnl': rep['pnl'],
                                        'sharpe': rep['metrics']['sharpe'], 'fitness': fit})
            elif rep.get('error'):
                print(f"    !! simulator error: {rep['error']}")

        round_json = {'round': k, 'teams': []}
        winner = None
        passing = [(t, e, r) for t, e, r, p in results if p]
        if passing:
            passing.sort(key=lambda x: (x[2]['metrics']['fitness'], x[2]['metrics']['sharpe']), reverse=True)
            winner = passing[0]
            for t, _, _, _ in results:
                if t is winner[0]:
                    t['wins'] += 1
        for team, expr, rep, passed in results:
            slim = {kk: vv for kk, vv in rep.items() if kk != 'pnl'}
            round_json['teams'].append({'team': team['name'], 'final_expression': expr,
                                        'passed': passed, 'report': slim})
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
            print("No pass — both teams failed all hard cutoffs this round.")
        self._print_leaderboard()

    def _print_report(self, rep):
        if rep.get('error'):
            return
        m = rep['metrics']
        print("    metrics:")
        print(f"      Sharpe {m['sharpe']:.2f} | Fitness {m['fitness']:.2f} | Returns {m['returns_pct']:.1f}% | "
              f"Turnover {m['turnover_pct']:.1f}% | Drawdown {m['drawdown_pct']:.1f}%")
        print(f"      Max weight {m['weight_concentration_pct']:.1f}% on {rep['max_weight_date']} | "
              f"Sub-universe Sharpe {rep['criteria']['subuniverse_sharpe']['value']:.2f} | "
              f"Self-corr {rep['criteria']['self_correlation']['value']:.2f}")
        print("    criteria:")
        for name, c in rep['criteria'].items():
            mark = 'PASS' if c['pass'] else 'FAIL'
            print(f"      [{mark}] {name}: {c['value']} (need {c['requirement']})")
        print("    yearly:")
        for y in rep['yearly']:
            print(f"      {y['year']}: Sharpe {y['sharpe']:+.2f} | Fitness {y['fitness']:.2f} | "
                  f"Turnover {y['turnover_pct']:.1f}% | Returns {y['returns_pct']:+.1f}%")

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
        else:
            champ = sorted(self.teams, key=lambda t: (-t['wins'], -(t['best_fitness'] or 0)))[0]
            print(f"Overall winner: {champ['name']} ({champ['wins']} round wins, "
                  f"best fitness {champ['best_fitness'] if champ['best_fitness'] is not None else 'n/a'})")
