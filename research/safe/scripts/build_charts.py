"""Build the SAFE web charts from the pinned public experiment results.

Usage: python3 research/safe/scripts/build_charts.py --cache /tmp/safe-interactive
Add --download to populate that cache from the public repository (stdlib only).
Raw prompts/responses stay outside the website; only aggregate counts are emitted.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen

COMMIT = 'a099eb8323e5d826cf6b8b4e4cefbfdab0e69d9e'
MODELS = ['gpt-5.5', 'o3', 'opus', 'sonnet']
NAMES = ['GPT-5.5', 'o3', 'Opus 4.8', 'Sonnet 4.6']
COLORS = ['#5E8B7E', '#4E79A7', '#D65F4A', '#7FA7D8']
PATHS = ['ablation/cot_faithfullness/utils.py',
         'ablation/cost_obligation_decomposition/scripts/utils.py']
for model in MODELS:
    PATHS += [f'main_experiment/coded/offered_report/{model}_offered_report.json',
              f'main_experiment/coded/offered_report/post_{model}_offered_report.json',
              f'ablation/cost_obligation_decomposition/results/{model}_cost_obligation_decomposition.json']
PATHS += ['ablation/cot_faithfullness/graphs/flip_rate_by_edit_type_counts.csv',
          'ablation/cot_faithfullness/data/faithfulness_metrics.jsonl',
          'ablation/cot_faithfullness/data/rule_based_acknowledgement.jsonl',
          'ablation/cot_faithfullness/data/llm_judge_acknowledgement.jsonl']


def point(k, n, ci=True):
    if not n:
        return dict(k=k, n=n, value=None, low=None, high=None)
    p = k / n
    z = 1.959963984540054
    center = (p + z*z/(2*n)) / (1+z*z/n)
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return dict(k=k, n=n, value=p*100,
                low=max(0, center-margin)*100 if ci else None,
                high=min(1, center+margin)*100 if ci else None)


def rate(rows, predicate, ci=True):
    return point(sum(bool(predicate(r)) for r in rows), len(rows), ci)


def series(name, color, values, **kw):
    return dict(name=name, color=color, values=values, **kw)


def panel(title, labels, series_list, note, kind='bar', **kw):
    return dict(title=title, labels=labels, series=series_list, note=note, kind=kind, **kw)


def majority(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r['scenario_id']].append(r)
    return [dict(runs[0], decision=Counter(r['decision'] for r in runs).most_common(1)[0][0])
            for runs in groups.values()]


def build(cache):
    load = lambda i: json.loads((cache / f'source-{i}').read_text())
    raw = [load(2+3*i) for i in range(4)]
    voted = [majority(rows) for rows in raw]
    charts = {}
    decisions = ['INSPECT', 'SKIP', 'UNCLEAR']
    decision_colors = ['#4E79A7', '#D65F4A', '#6B7280']
    charts['inspection-rates'] = [
        panel(title, NAMES, [series(d.title(), c, [rate(rows, lambda r: r['decision'] == d) for rows in data])
                            for d, c in zip(decisions, decision_colors)],
              'Offered Report · '+unit+' · 95% Wilson intervals.')
        for title, data, unit in [('Scenario majority vote', voted, 'one vote per scenario'),
                                  ('Individual rollouts', raw, 'all individual rollouts')]]
    charts['figure-2'] = []
    for field, levels, labels, axis in [
        ('severity_label', ['negligible', 'low', 'medium', 'high'], ['Negligible', 'Low', 'Medium', 'High'], 'Severity'),
        ('probability_label', ['low', 'medium', 'high'], ['10%', '40%', '70%'], 'Probability')]:
        for code, title in [('INSPECT', 'Inspection'), ('A', 'Halt'), ('B', 'Proceed'), ('C', 'Partial mitigation')]:
            all_series = []
            for name, color, rows in zip(NAMES, COLORS, raw):
                values = []
                for level in levels:
                    eligible = [r for r in rows if r[field] == level and
                                (code == 'INSPECT' or (r['decision'] == 'INSPECT' and r.get('post_inspection_decision') in ('A', 'B', 'C')))]
                    values.append(rate(eligible, lambda r: r['decision' if code == 'INSPECT' else 'post_inspection_decision'] == code))
                all_series.append(series(name, color, values))
            charts['figure-2'].append(panel(f'{title} by {axis.lower()}', labels, all_series,
                ('All rollouts' if code == 'INSPECT' else 'Inspected cases with a valid A/B/C action')+' · 95% Wilson intervals.', 'line'))
    costs = ['free', 'low', 'medium', 'high', 'extreme']
    charts['figure-3'] = [panel(name, ['Free (0)', 'Low (10)', 'Medium (30)', 'High (60)', 'Procedural'],
        [series(d.title(), c, [rate([r for r in rows if r['cost_label'] == cost], lambda r: r['decision'] == d) for cost in costs])
         for d, c in zip(decisions, decision_colors)],
        'Scenario majority votes · procedural “extreme” condition is separate from numeric costs · 95% Wilson intervals.',
        'line', separateLast=True) for name, rows in zip(NAMES, voted)]
    actions = ['Halt', 'Proceed', 'Partial mitigation']
    action_colors = ['#4E79A7', '#D65F4A', '#5E8B7E']
    charts['figure-4'] = [panel('Post-inspection actions', NAMES,
        [series(name, color, [rate([r for r in rows if r.get('post_inspection_decision') in ('A', 'B', 'C')],
                                      lambda r: r['post_inspection_decision'] == code) for rows in raw])
         for code, name, color in zip('ABC', actions, action_colors)],
        'Valid post-inspection A/B/C actions · 95% Wilson intervals.')]
    categories = [('Halt-required: halt', {'medium', 'high'}, 'A'), ('Negligible: proceed', {'negligible'}, 'B'),
                  ('Low: partial mitigation', {'low'}, 'C')]
    charts['figure-4'].append(panel('Agreement with expected action', NAMES,
        [series(name, color, [rate([r for r in rows if r['decision'] == 'INSPECT' and r['severity_label'] in levels and
                                           r.get('post_inspection_decision') in ('A', 'B', 'C')],
                                      lambda r: r['post_inspection_decision'] == code) for rows in raw])
         for (name, levels, code), color in zip(categories, action_colors)],
        'Inspected cases within each severity category · 95% Wilson intervals.'))
    conditions = ['free_no_obligation', 'cost_only', 'obligation_only', 'cost_plus_obligation',
                  'obligation_only_no_cancel', 'cost_plus_obligation_no_cancel']
    condition_labels = ['Free / no obligation', 'Cost only', 'Obligation / cancel', 'Cost + obligation / cancel',
                        'Obligation / no cancel', 'Cost + obligation / no cancel']
    ablation = [load(4+3*i) for i in range(4)]
    charts['figure-6'] = []
    for indices, provider in [([0, 1], 'OpenAI'), ([2, 3], 'Anthropic')]:
        charts['figure-6'].append(panel(f'{provider} · condition rates', condition_labels,
            [series(NAMES[i], COLORS[i], [rate([r for r in ablation[i] if r['condition'] == cond and r['decision'] in ('INSPECT', 'SKIP')],
                                               lambda r: r['decision'] == 'INSPECT') for cond in conditions]) for i in indices],
            'Valid INSPECT/SKIP rollouts in 50 selected scenarios · 95% Wilson intervals.'))
    effects = [('Cost', 'cost_only', 'free_no_obligation'),
               ('Cost + cancel', 'cost_plus_obligation', 'obligation_only'),
               ('Obligation (costly)', 'cost_plus_obligation', 'cost_only'),
               ('Cancellation (free)', 'obligation_only', 'obligation_only_no_cancel'),
               ('Cancellation (costly)', 'cost_plus_obligation', 'cost_plus_obligation_no_cancel')]
    for name, rows in zip(NAMES, ablation):
        groups = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r['decision'] in ('INSPECT', 'SKIP'):
                groups[r['scenario_id']][r['condition']].append(r['decision'] == 'INSPECT')
        matched = [{c: sum(v)/len(v) for c, v in g.items()} for g in groups.values() if all(c in g for c in conditions)]
        values = [[], [], []]
        for _, a, b in effects:
            differences = [r[a]-r[b] for r in matched]
            for i, pred in enumerate([lambda d: d < 0, lambda d: d == 0, lambda d: d > 0]):
                values[i].append(rate(differences, pred, ci=False))
        charts['figure-6'].append(panel(f'{name} · matched scenario changes', [e[0] for e in effects],
            [series(n, c, v) for n, c, v in zip(['Lower inspection', 'No change', 'Higher inspection'],
                                               ['#B54735', '#6B7280', '#5E8B7E'], values)],
            'Direction of within-scenario inspection-rate change · complete six-condition matches · selected scenarios.', 'stacked'))
    edits = ['cost_removed', 'severity_raised', 'probability_raised', 'obligation_removed',
             'evidence_made_report_shaped', 'finding_made_harder_to_rationalize']
    edit_labels = ['Cost removed', 'Severity raised', 'Probability raised', 'Obligation removed', 'Report framing', 'Reduced mitigability']
    counts = list(csv.DictReader((cache/'source-14').open()))
    charts['figure-7'] = []
    for key, labels, categories in [('edit_type', edit_labels, edits),
            ('rationale_label', ['Expected value', 'Performative mitigation', 'Rationalization'],
             ['expected_value', 'performative_mitigation', 'rationalization'])]:
        all_series = []
        for m, name, color in zip(MODELS, NAMES, COLORS):
            points = []
            for category in categories:
                row = next(r for r in counts if r['panel'] == key and r['model'] == m and r['category'] == category)
                p = point(int(row['k']), int(row['n']))
                if p['n']:
                    assert abs(p['value']/100-float(row['rate'])) < 1e-12
                    assert abs(p['low']/100-float(row['ci_low'])) < 1e-12
                points.append(p)
            all_series.append(series(name, color, points))
        charts['figure-7'].append(panel('Predicted-direction changes by '+('intervention' if key == 'edit_type' else 'original rationale'),
            labels, all_series, 'Parsed diagnostic cases · 95% Wilson intervals · not population treatment effects. Opus has no eligible obligation-removal cases.'))
    metrics = [json.loads(line) for line in (cache/'source-15').read_text().splitlines() if line.strip()]
    ack_series = []
    for m, name, color in zip(MODELS, NAMES, COLORS):
        for scorer, field, marker in [('Rule-based', 'acknowledged_by_keyword', 'circle'),
                                      ('Gemini', 'acknowledged_by_llm', 'diamond')]:
            values = [rate([r for r in metrics if r['model'] == m and r['edit_type'] == e and r['parse_status'] == 'ok'],
                           lambda r: bool(r.get(field)), ci=False) for e in edits]
            ack_series.append(series(f'{name} · {scorer}', color, values, marker=marker, model=name))
    charts['figure-8'] = [panel('Rationale acknowledgement of the edited factor', edit_labels, ack_series,
        'Parsed diagnostic rationales · circles: rule-based; diamonds: Gemini · no eligible Opus obligation-removal cases.', 'paired')]
    return dict(commit=COMMIT, charts=charts,
                sources=[dict(path=p, sha256=hashlib.sha256((cache/f'source-{i}').read_bytes()).hexdigest()) for i, p in enumerate(PATHS)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    if args.download:
        for i, path in enumerate(PATHS):
            dest = args.cache / f'source-{i}'
            if not dest.exists():
                with urlopen(f'https://raw.githubusercontent.com/omertafveez-2001/Do-Models-Seek-Safety-Evidence/{COMMIT}/{path}') as response:
                    dest.write_bytes(response.read())
    data = build(args.cache)
    output = Path(__file__).resolve().parents[1] / 'charts-data.js'
    output.write_text('/* Generated by scripts/build_charts.py; source commit '+COMMIT+' */\nwindow.SAFE_CHARTS = '+json.dumps(data, separators=(',', ':'), allow_nan=False)+';\n')
    print(f'Built {len(data["charts"])} figures / {sum(map(len, data["charts"].values()))} panels: {output}')
