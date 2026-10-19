/* Native SVG charts: no runtime dependencies or remote data requests. */
(() => {
    'use strict';
    const data = window.SAFE_CHARTS;
    if (!data) return;
    const ns = 'http://www.w3.org/2000/svg';
    const fmt = value => value == null ? 'No eligible cases' : `${value.toFixed(1)}%`;
    const el = (tag, attrs = {}, text) => {
        const node = document.createElement(tag);
        for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
        if (text != null) node.textContent = text;
        return node;
    };
    const svgEl = (tag, attrs = {}, text) => {
        const node = document.createElementNS(ns, tag);
        for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
        if (text != null) node.textContent = text;
        return node;
    };
    function describe(s, label, p) {
        return `${s.name} · ${label}: ${fmt(p.value)}${p.n ? ` (${p.k}/${p.n})` : ''}` +
            (p.low == null ? '' : ` · 95% Wilson interval ${fmt(p.low)}–${fmt(p.high)}`);
    }
    function downloadCSV(id, panels) {
        const rows = [['Panel', 'Category', 'Series', 'Count', 'Denominator', 'Percent', '95% Wilson lower', '95% Wilson upper']];
        for (const p of panels) for (const s of p.series) s.values.forEach((v, i) => {
            rows.push([p.title, p.labels[i], s.name, v.k, v.n, v.value, v.low, v.high]);
        });
        const csv = rows.map(row => row.map(v => `"${String(v ?? '').replaceAll('"', '""')}"`).join(',')).join('\r\n');
        const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
        const a = el('a', { href: url, download: `safe-${id}.csv` });
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
    function wrapLabel(svg, label, x, y, maxChars) {
        const text = svgEl('text', { x, y, 'text-anchor': 'middle', class: 'chart-axis-label' });
        const lines = [];
        for (const word of label.split(' ')) {
            if (!lines.length || lines.at(-1).length + word.length + 1 > maxChars) lines.push(word);
            else lines[lines.length - 1] += ` ${word}`;
        }
        lines.forEach((line, i) => text.append(svgEl('tspan', { x, dy: i ? 15 : 0 }, line)));
        svg.append(text);
    }
    function createPanel(panel, id, index, intervals) {
        const card = el('section', { class: 'chart-panel', 'aria-labelledby': `${id}-panel-${index}` });
        if (panel.labels.length > 4 && panel.kind !== 'line') card.classList.add('chart-panel-wide');
        card.append(el('h3', { id: `${id}-panel-${index}` }, panel.title));
        const plot = el('div', { class: 'chart-plot' });
        const tooltip = el('div', { class: 'chart-tooltip', role: 'status', hidden: '' });
        const legend = el('div', { class: 'chart-legend', 'aria-label': `${panel.title}: toggle series` });
        const active = new Set(panel.series.map(s => s.name));
        panel.series.forEach(s => {
            const button = el('button', { type: 'button', 'aria-pressed': 'true', class: 'chart-series' });
            const swatch = el('span', { class: `chart-swatch ${s.marker || ''}`, 'aria-hidden': 'true' });
            swatch.style.backgroundColor = s.color;
            button.append(swatch, document.createTextNode(s.name));
            button.addEventListener('click', () => {
                if (active.has(s.name)) active.delete(s.name); else active.add(s.name);
                button.setAttribute('aria-pressed', String(active.has(s.name)));
                draw();
            });
            legend.append(button);
        });
        const note = el('p', { class: 'chart-note' }, panel.note);
        card.append(plot, legend, note);
        function draw() {
            tooltip.hidden = true;
            const svg = svgEl('svg', { viewBox: '0 0 520 370', class: 'interactive-svg',
                role: 'group', 'aria-label': `${panel.title}. Percentage axis from 0 to 100. Use the series buttons to compare; tab to data points or open the data table.` });
            svg.append(svgEl('title', {}, panel.title));
            const left = 48, top = 22, width = 450, height = 240;
            const y = value => top + height * (1 - value / 100);
            const step = width / panel.labels.length;
            const x = i => left + step * (i + 0.5);
            [0, 25, 50, 75, 100].forEach(value => {
                svg.append(svgEl('line', { x1: left, x2: left + width, y1: y(value), y2: y(value), class: 'chart-gridline' }));
                svg.append(svgEl('text', { x: left - 8, y: y(value) + 4, 'text-anchor': 'end', class: 'chart-axis-label' }, `${value}%`));
            });
            if (panel.separateLast) {
                const divider = x(panel.labels.length - 1) - step / 2;
                svg.append(svgEl('line', { x1: divider, x2: divider, y1: top, y2: top + height, class: 'chart-procedural' }));
            }
            panel.labels.forEach((label, i) => wrapLabel(svg, label, x(i), y(0) + 23, Math.max(10, Math.floor(step / 7))));
            const visible = panel.series.filter(s => active.has(s.name));
            const totals = panel.labels.map(() => 0);
            const barWidth = Math.min(32, step * 0.76 / Math.max(visible.length, 1));
            const pairedModels = [...new Set(visible.map(s => s.model))];
            const pairedWidth = Math.min(24, step * 0.7 / Math.max(pairedModels.length, 1));
            if (panel.kind === 'paired') {
                for (const model of pairedModels) {
                    const pair = visible.filter(s => s.model === model);
                    if (pair.length !== 2) continue;
                    panel.labels.forEach((_, i) => {
                        const a = pair[0].values[i], b = pair[1].values[i];
                        if (a.value == null || b.value == null) return;
                        const px = x(i) + (pairedModels.indexOf(model) - (pairedModels.length - 1) / 2) * pairedWidth;
                        svg.append(svgEl('line', { x1: px - 2, x2: px + 2, y1: y(a.value), y2: y(b.value),
                            stroke: pair[0].color, 'stroke-width': 2, 'pointer-events': 'none' }));
                    });
                }
            }
            function mark(node, s, i, p) {
                const description = describe(s, panel.labels[i], p);
                node.setAttribute('tabindex', '0');
                node.setAttribute('role', 'img');
                node.setAttribute('aria-label', description);
                node.classList.add('chart-mark');
                node.append(svgEl('title', {}, description));
                const show = () => {
                    tooltip.textContent = description;
                    tooltip.hidden = false;
                };
                node.addEventListener('pointerenter', show);
                node.addEventListener('pointerleave', () => { tooltip.hidden = true; });
                node.addEventListener('focus', show);
                node.addEventListener('blur', () => { tooltip.hidden = true; });
                node.addEventListener('click', show);
                node.addEventListener('keydown', e => { if (e.key === 'Escape') tooltip.hidden = true; });
                svg.append(node);
            }
            visible.forEach((s, si) => {
                const offset = (si - (visible.length - 1) / 2) * barWidth;
                const coords = s.values.map((p, i) => {
                    const px = x(i) + (panel.kind === 'paired'
                        ? (pairedModels.indexOf(s.model) - (pairedModels.length - 1) / 2) * pairedWidth + (s.marker === 'diamond' ? 2 : -2)
                        : panel.kind === 'line' || panel.kind === 'stacked' ? 0 : offset);
                    return { p, i, px, py: p.value == null ? null : y(p.value) };
                });
                if (panel.kind === 'line') {
                    let segment = [];
                    const flush = () => {
                        if (segment.length > 1) svg.append(svgEl('polyline', { points: segment.join(' '), fill: 'none',
                            stroke: s.color, 'stroke-width': 2.5, 'stroke-dasharray': si === 1 ? '7 3' : si === 2 ? '2 3' : '' }));
                        segment = [];
                    };
                    coords.forEach(({ p, i, px, py }) => {
                        if (py == null || (panel.separateLast && i === coords.length - 1)) flush();
                        if (py != null) segment.push(`${px},${py}`);
                    });
                    flush();
                }
                coords.forEach(({ p, i, px, py }) => {
                    if (py == null) return;
                    if (panel.kind === 'bar' || panel.kind === 'stacked') {
                        const stacked = panel.kind === 'stacked';
                        const base = stacked ? totals[i] : 0;
                        const bw = stacked ? step * 0.58 : barWidth * 0.88;
                        mark(svgEl('rect', { x: px - bw / 2, y: y(base + p.value), width: bw,
                            height: Math.max(1.5, height * p.value / 100), fill: s.color,
                            stroke: 'white', 'stroke-width': 0.6 }), s, i, p);
                        if (stacked) totals[i] += p.value;
                    }
                    if (intervals.checked && p.low != null) {
                        const g = svgEl('g', { stroke: s.color, 'stroke-width': 1.3, 'pointer-events': 'none' });
                        g.append(svgEl('line', { x1: px, x2: px, y1: y(p.low), y2: y(p.high) }));
                        for (const bound of [p.low, p.high]) g.append(svgEl('line', { x1: px - 3, x2: px + 3, y1: y(bound), y2: y(bound) }));
                        svg.append(g);
                    }
                    if (panel.kind === 'line' || panel.kind === 'paired') {
                        const shape = s.marker === 'diamond'
                            ? svgEl('path', { d: `M ${px} ${py-6} l 6 6 l -6 6 l -6 -6 Z`, fill: s.color, stroke: 'white' })
                            : svgEl('circle', { cx: px, cy: py, r: 5.5, fill: s.color, stroke: 'white', 'stroke-width': 1.5 });
                        mark(shape, s, i, p);
                    }
                });
            });
            if (!visible.length) svg.append(svgEl('text', { x: 273, y: 140, 'text-anchor': 'middle', class: 'chart-empty' }, 'Select a series below to show its data.'));
            plot.replaceChildren(svg, tooltip);
        }
        intervals.addEventListener('change', draw);
        draw();
        return card;
    }
    for (const [id, panels] of Object.entries(data.charts)) {
        const figure = document.getElementById(id);
        const fallback = figure?.querySelector('.figure-image');
        if (!fallback) continue;
        try {
            const root = el('div', { class: 'interactive-figure' });
            const toolbar = el('div', { class: 'chart-toolbar' });
            const viewLabel = el('label', { for: `${id}-view` }, 'View');
            const view = el('select', { id: `${id}-view` });
            view.append(el('option', { value: 'all' }, 'All panels'));
            panels.forEach((p, i) => view.append(el('option', { value: String(i) }, p.title)));
            const intervalsLabel = el('label', { class: 'chart-intervals' });
            const intervals = el('input', { type: 'checkbox', checked: '' });
            intervalsLabel.append(intervals, document.createTextNode('95% intervals'));
            if (!panels.some(p => p.series.some(s => s.values.some(v => v.low != null)))) intervalsLabel.hidden = true;
            const download = el('button', { type: 'button', class: 'chart-download' }, 'Download CSV ↓');
            download.addEventListener('click', () => downloadCSV(id, panels));
            toolbar.append(viewLabel, view, intervalsLabel, download);
            const hint = el('p', { class: 'chart-hint' }, 'Hover, tap, or focus a mark for values. Click a legend item to show or hide it.');
            const grid = el('div', { class: 'chart-panels' });
            const cards = panels.map((p, i) => createPanel(p, id, i, intervals));
            grid.append(...cards);
            view.addEventListener('change', () => {
                cards.forEach((card, i) => { card.hidden = view.value !== 'all' && Number(view.value) !== i; });
                grid.classList.toggle('single-panel', view.value !== 'all' || cards.length === 1);
            });
            if (cards.length === 1) grid.classList.add('single-panel');
            const details = el('details', { class: 'chart-data' });
            details.append(el('summary', {}, 'View data table'));
            for (const p of panels) {
                const wrap = el('div', { class: 'chart-table-wrap', tabindex: '0', role: 'region', 'aria-label': `${p.title} data` });
                const table = el('table');
                table.append(el('caption', {}, p.title));
                const head = el('thead'), header = el('tr');
                ['Category', 'Series', 'Rate', 'Count / total', '95% Wilson interval'].forEach(t => header.append(el('th', { scope: 'col' }, t)));
                head.append(header);
                const body = el('tbody');
                for (const s of p.series) s.values.forEach((v, i) => {
                    const row = el('tr');
                    row.append(el('th', { scope: 'row' }, p.labels[i]));
                    [s.name, fmt(v.value), `${v.k} / ${v.n}`, v.low == null ? '—' : `${fmt(v.low)}–${fmt(v.high)}`]
                        .forEach(t => row.append(el('td', {}, t)));
                    body.append(row);
                });
                table.append(head, body);
                wrap.append(table);
                details.append(wrap);
            }
            root.append(toolbar, hint, grid, details);
            figure.insertBefore(root, fallback);
            // Keep the paper render available, including when scripting is disabled.
            const original = el('details', { class: 'chart-original' });
            original.append(el('summary', {}, 'View original paper graphic'));
            fallback.replaceWith(original);
            original.append(fallback);
        } catch (error) {
            console.error(`Unable to render ${id}; retaining original graphic.`, error);
        }
    }
})();
